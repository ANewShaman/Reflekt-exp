"""
Reflekt Visual Engine – Hybrid (30% Water / 70% Organism)
--------------------------------------------------------
Requirements:
    pip install pygame numpy

Features:
 - Organism-first visuals (branching/growing nodes) driven by valence/arousal.
 - Light-weight fluid background (low-res dye advection + diffusion) for ink/water behavior.
 - seed_word(word) to create organism seeds & inject colored dye at runtime (call from speech-recognizer).
 - Hooks to read from emotion engine: engine.latest_frame and engine.voice_last (if provided).
 - Performance knobs at top of file.

Usage (standalone test):
    python reflekt_visuals.py

Usage (with your engine/reflekt_main.py):
    from reflekt_visuals import ReflektVisualEngine
    vis = ReflektVisualEngine(engine=your_engine, config={...})
    # run in background thread:
    import threading
    t = threading.Thread(target=vis.run, daemon=True)
    t.start()
"""

import pygame
import numpy as np
import time
import math
import random
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# -----------------------
# CONFIG / PERFORMANCE
# -----------------------
CONFIG = {
    "width": 1280,
    "height": 720,
    "fps": 45,                    # Lower than 60 to reduce load on speech recog
    "fluid_res_factor": 6,        # 1 => full res (slow), larger => lower res (faster)
    "max_organisms": 80,
    "organism_growth_rate": 1.0,  # base multiplier for growth
    "particle_limit": 800,
    "ink_alpha": 160,             # alpha for dye overlay (0..255)
    "organism_opacity": 200,
    "auto_seed_interval": 2.5,    # Auto-seed every N seconds in demo mode
}

# Emotion palettes (same-ish as earlier, pared)
PALETTES = {
    "happy": (255, 200, 120),
    "sad": (65, 95, 200),
    "angry": (200, 60, 50),
    "neutral": (160, 150, 170),
    "surprise": (255, 160, 200),
    "fear": (140, 110, 180),
    "disgust": (110, 170, 90),
}

def clamp(v, a, b): return max(a, min(b, v))

# -----------------------
# Fluid (low-res dye + simple advection/diffuse)
# -----------------------
class FluidField:
    """Simple low-res fluid dye field. Very lightweight."""
    def __init__(self, width:int, height:int, res_factor:int):
        self.width = width
        self.height = height
        self.res_w = max(8, width // res_factor)
        self.res_h = max(6, height // res_factor)
        self.dye = np.zeros((self.res_h, self.res_w, 3), dtype=np.float32)
        self.velocity = np.zeros((self.res_h, self.res_w, 2), dtype=np.float32)
        self.tmp = np.zeros_like(self.dye)
        self.decay = 0.985
        self.visc = 0.25

    def step(self, dt:float):
        """Advect dye by velocity (semi-Lagrangian-ish) and blur a bit."""
        # advect
        h, w = self.res_h, self.res_w
        ys, xs = np.mgrid[0:h, 0:w]
        # compute source coords
        src_x = (xs - self.velocity[...,0] * dt).astype(np.float32)
        src_y = (ys - self.velocity[...,1] * dt).astype(np.float32)
        # wrap (toroidal) for smooth visuals
        src_x = np.mod(src_x, w)
        src_y = np.mod(src_y, h)
        # bilinear sample
        x0 = np.floor(src_x).astype(int); x1 = (x0 + 1) % w
        y0 = np.floor(src_y).astype(int); y1 = (y0 + 1) % h
        fx = src_x - x0; fy = src_y - y0
        for c in range(3):
            v00 = self.dye[y0, x0, c]
            v10 = self.dye[y0, x1, c]
            v01 = self.dye[y1, x0, c]
            v11 = self.dye[y1, x1, c]
            interp = (v00 * (1 - fx) * (1 - fy) +
                      v10 * fx * (1 - fy) +
                      v01 * (1 - fx) * fy +
                      v11 * fx * fy)
            self.tmp[..., c] = interp
        # simple diffusion (blur)
        self.tmp = self.tmp * (1 - self.visc) + (
            (np.roll(self.tmp, 1, axis=0) + np.roll(self.tmp, -1, axis=0) +
             np.roll(self.tmp, 1, axis=1) + np.roll(self.tmp, -1, axis=1)) * (self.visc / 4.0)
        )
        # decay
        self.tmp *= self.decay
        self.dye[:] = self.tmp

        # damp velocities
        self.velocity *= 0.98

    def splat_dye(self, x_px:int, y_px:int, color:Tuple[int,int,int], strength:float=1.0, radius_px:int=40):
        """Add dye at pixel coords (screen space)."""
        # convert to low-res coords
        rx = int((x_px / self.width) * self.res_w)
        ry = int((y_px / self.height) * self.res_h)
        r = max(1, int(radius_px / max(1, self.width//self.res_w)))
        y0 = clamp(ry - r, 0, self.res_h-1); y1 = clamp(ry + r, 0, self.res_h-1)
        x0 = clamp(rx - r, 0, self.res_w-1); x1 = clamp(rx + r, 0, self.res_w-1)
        col = np.array(color, dtype=np.float32)
        for yy in range(y0, y1+1):
            for xx in range(x0, x1+1):
                dist = math.hypot(xx - rx, yy - ry)
                if dist <= r:
                    fall = (1.0 - (dist / (r+1))) * strength
                    self.dye[yy, xx] = np.maximum(self.dye[yy, xx], col * fall)

    def add_force(self, x_px:int, y_px:int, fx:float, fy:float, radius_px:int=50):
        rx = int((x_px / self.width) * self.res_w)
        ry = int((y_px / self.height) * self.res_h)
        r = max(1, int(radius_px / max(1, self.width//self.res_w)))
        y0 = clamp(ry - r, 0, self.res_h-1); y1 = clamp(ry + r, 0, self.res_h-1)
        x0 = clamp(rx - r, 0, self.res_w-1); x1 = clamp(rx + r, 0, self.res_w-1)
        for yy in range(y0, y1+1):
            for xx in range(x0, x1+1):
                dist = math.hypot(xx - rx, yy - ry)
                if dist <= r:
                    fall = (1.0 - (dist / (r+1)))
                    self.velocity[yy, xx, 0] += fx * fall
                    self.velocity[yy, xx, 1] += fy * fall

    def render_surface(self):
        """Return a pygame surface upscaled to screen size with alpha."""
        arr = np.clip(self.dye, 0, 255).astype(np.uint8)
        # transpose because pygame expects (w,h,3) array with x first via surfarray
        surf = pygame.surfarray.make_surface(np.transpose(arr, (1,0,2)))
        scaled = pygame.transform.smoothscale(surf, (self.width, self.height))
        scaled.set_alpha(CONFIG['ink_alpha'])
        return scaled

# -----------------------
# Organisms: node/branch growth system (dominant, organism-first)
# -----------------------
@dataclass
class Node:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float
    hue: Tuple[int,int,int]  # RGB
    shape_type: str  # "petal", "spike", "web", "wave", "star"
    rotation: float

class Organism:
    """A single organism: spawns from a seed word or splash, grows branches."""
    def __init__(self, x:int, y:int, color:Tuple[int,int,int], valence:float, arousal:float):
        self.nodes: List[Node] = []
        self.age = 0.0
        self.growth = 1.0 + (arousal * 2.0)      # growth rate influenced by arousal
        self.curvature = 0.4 + (1.0 - valence) * 0.6  # valence shifts shape curvature
        self.color = color
        
        # Choose organism shape based on emotion
        # High arousal → spiky/energetic (spikes, stars)
        # Low arousal → soft/flowing (petals, waves)
        # High valence → organic/pleasant (petals, stars)
        # Low valence → harsh/geometric (spikes, webs)
        if arousal > 0.6:
            if valence > 0.3:
                self.shape_type = random.choice(["star", "petal"])  # excited, happy
            else:
                self.shape_type = random.choice(["spike", "web"])  # angry, fearful
        else:
            if valence > 0.3:
                self.shape_type = random.choice(["petal", "wave"])  # calm, peaceful
            else:
                self.shape_type = random.choice(["wave", "web"])  # sad, melancholic
        
        # initial node (root)
        n = Node(x=float(x), y=float(y), vx=0.0, vy=-0.2 * self.growth, life=3.0 + self.growth*2.0,
                 size=4.0 + self.growth*3.0, hue=color, shape_type=self.shape_type, rotation=0.0)
        self.nodes.append(n)
        self.max_nodes = int(40 + self.growth * 30)

    def update(self, dt:float, flow_func):
        """Update nodes: grow new nodes, apply flow field forces, prune."""
        self.age += dt
        new_nodes = []
        for n in self.nodes:
            # flow forces: give a slight bias from flow field
            fx, fy = flow_func(n.x, n.y)
            # curvature noise: turns influenced by curvature
            angle = math.atan2(n.vy, n.vx) + (random.uniform(-1,1) * 0.15 * self.curvature)
            speed = math.hypot(n.vx, n.vy) * 0.98 + 0.1 * self.growth
            n.vx = math.cos(angle) * speed + fx * 4.0 * dt
            n.vy = math.sin(angle) * speed + fy * 4.0 * dt
            # position
            n.x += n.vx * dt * 60.0
            n.y += n.vy * dt * 60.0
            n.life -= dt * (0.5 + 0.5 * self.growth)
            n.size = max(0.5, n.size * (0.995))
            # rotate nodes for dynamic shapes
            n.rotation += dt * (0.5 + self.growth * 0.3)
            
            # spawn new node occasionally
            if len(self.nodes) + len(new_nodes) < self.max_nodes and random.random() < (0.02 + 0.03 * self.growth):
                # branch
                nx = n.x + n.vx * 6.0 + random.uniform(-4,4)
                ny = n.y + n.vy * 6.0 + random.uniform(-4,4)
                nvx = n.vx * 0.8 + random.uniform(-0.2, 0.2)
                nvy = n.vy * 0.8 + random.uniform(-0.2, 0.2)
                nn = Node(x=nx, y=ny, vx=nvx, vy=nvy, life=3.0 + self.growth*1.5, size=n.size*0.9, 
                         hue=self.color, shape_type=self.shape_type, rotation=n.rotation)
                new_nodes.append(nn)
        # append new nodes
        self.nodes.extend(new_nodes)
        # prune dead nodes
        self.nodes = [n for n in self.nodes if n.life > 0 and  -2000 < n.x < 2000 and -2000 < n.y < 2000]

    def is_dead(self):
        return len(self.nodes) == 0

    def draw(self, surface:pygame.Surface, opacity:int=200):
        """Draw organism with defined abstract shapes based on emotion."""
        if not self.nodes: return
        
        col = self.color
        
        # Draw each node with its specific shape
        for n in self.nodes:
            alpha = clamp(int(opacity * (n.life / 4.0)), 10, opacity)
            s = max(2, int(n.size))
            
            # Create surface for this node
            surf_size = s * 6
            node_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            center = surf_size // 2
            
            # Draw shape based on type
            if n.shape_type == "petal":
                # Flower petal - teardrop shape
                for i in range(6):  # 6 petals
                    angle = (i * 60 + n.rotation * 20) * math.pi / 180
                    # Draw elongated ellipse for petal
                    petal_surf = pygame.Surface((s*3, s*5), pygame.SRCALPHA)
                    pygame.draw.ellipse(petal_surf, (*col, alpha), (0, 0, s*3, s*5))
                    # Rotate and position petal
                    rotated = pygame.transform.rotate(petal_surf, -angle * 180 / math.pi)
                    offset_x = center + int(math.cos(angle) * s * 1.5) - rotated.get_width()//2
                    offset_y = center + int(math.sin(angle) * s * 1.5) - rotated.get_height()//2
                    node_surf.blit(rotated, (offset_x, offset_y), special_flags=pygame.BLEND_ALPHA_SDL2)
                # Center circle
                pygame.draw.circle(node_surf, (*col, alpha), (center, center), s)
                
            elif n.shape_type == "spike":
                # Sharp spikes radiating outward
                for i in range(8):  # 8 spikes
                    angle = (i * 45 + n.rotation * 30) * math.pi / 180
                    # Draw triangle spike
                    points = [
                        (center, center),
                        (center + int(math.cos(angle - 0.2) * s * 3), 
                         center + int(math.sin(angle - 0.2) * s * 3)),
                        (center + int(math.cos(angle + 0.2) * s * 3), 
                         center + int(math.sin(angle + 0.2) * s * 3))
                    ]
                    pygame.draw.polygon(node_surf, (*col, alpha), points)
                # Core circle
                pygame.draw.circle(node_surf, (*col, alpha), (center, center), s)
                
            elif n.shape_type == "wave":
                # Flowing wave pattern
                points = []
                for i in range(12):
                    angle = (i * 30 + n.rotation * 15) * math.pi / 180
                    wave_offset = math.sin(angle * 3) * s * 0.5
                    radius = s * 2 + wave_offset
                    points.append((
                        center + int(math.cos(angle) * radius),
                        center + int(math.sin(angle) * radius)
                    ))
                if len(points) > 2:
                    pygame.draw.polygon(node_surf, (*col, alpha), points)
                # Smooth edges with circle overlay
                pygame.draw.circle(node_surf, (*col, alpha//2), (center, center), s)
                
            elif n.shape_type == "web":
                # Geometric web/mandala pattern
                num_points = 6
                for ring in range(3):
                    ring_radius = s * (1 + ring * 0.8)
                    for i in range(num_points):
                        angle1 = (i * 360/num_points + n.rotation * 25) * math.pi / 180
                        angle2 = ((i+1) * 360/num_points + n.rotation * 25) * math.pi / 180
                        p1 = (center + int(math.cos(angle1) * ring_radius),
                              center + int(math.sin(angle1) * ring_radius))
                        p2 = (center + int(math.cos(angle2) * ring_radius),
                              center + int(math.sin(angle2) * ring_radius))
                        pygame.draw.line(node_surf, (*col, alpha), p1, p2, max(1, s//3))
                        # Connect to center
                        pygame.draw.line(node_surf, (*col, alpha//2), (center, center), p1, max(1, s//4))
                pygame.draw.circle(node_surf, (*col, alpha), (center, center), s//2)
                
            elif n.shape_type == "star":
                # Multi-pointed star
                num_points = 5
                points = []
                for i in range(num_points * 2):
                    angle = (i * 180/num_points + n.rotation * 20) * math.pi / 180
                    if i % 2 == 0:
                        radius = s * 2.5  # outer points
                    else:
                        radius = s * 1.2  # inner points
                    points.append((
                        center + int(math.cos(angle) * radius),
                        center + int(math.sin(angle) * radius)
                    ))
                if len(points) > 2:
                    pygame.draw.polygon(node_surf, (*col, alpha), points)
                pygame.draw.circle(node_surf, (*col, alpha), (center, center), s//2)
            
            # Blit the node to main surface with additive blending
            surface.blit(node_surf, (int(n.x - surf_size//2), int(n.y - surf_size//2)), 
                        special_flags=pygame.BLEND_ADD)
        
        # Connect nodes with flowing lines (organic connection)
        if len(self.nodes) > 1:
            # Draw connections between nearby nodes
            for i, n1 in enumerate(self.nodes):
                for n2 in self.nodes[i+1:i+4]:  # Connect to next 3 nodes max
                    dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
                    if dist < 100:  # Only connect close nodes
                        # Fade line based on distance
                        line_alpha = int(150 * (1 - dist/100))
                        if line_alpha > 20:
                            pygame.draw.aaline(surface, (*col, line_alpha), 
                                             (int(n1.x), int(n1.y)), 
                                             (int(n2.x), int(n2.y)), blend=1)

# -----------------------
# Hybrid Visual Engine
# -----------------------
class ReflektVisualEngine:
    """Main visual engine class (organism-forward, with fluid background)."""
    def __init__(self, engine=None, config:Dict=None):
        self.engine = engine
        self.config = {**CONFIG, **(config or {})}
        self.W = self.config["width"]; self.H = self.config["height"]
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Reflekt – Hybrid Visuals (30% Water / 70% Organism)")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.running = False

        # fluid
        self.fluid = FluidField(self.W, self.H, self.config["fluid_res_factor"])

        # organisms list
        self.organisms: List[Organism] = []

        # particles for accent (tiny glowing motes)
        self.particles: List[Dict] = []

        # word seed buffer (thread-safe-ish)
        self._seed_queue: List[Tuple[str, float]] = []

        # demo fallback state
        self.demo_t = 0.0
        self.last_auto_seed = 0.0
        
        print("✓ Visual engine initialized")
        print(f"  Resolution: {self.W}x{self.H}")
        print(f"  Target FPS: {self.config['fps']}")
        print(f"  Fluid resolution: {self.fluid.res_w}x{self.fluid.res_h}")

    # -------------------
    # External hook: call from speech recognizer or main when a word occurs
    # -------------------
    def seed_word(self, word:str, intensity:float=1.0):
        """Public API: seed a visual organism and dye drop from a recognized word."""
        # store to queue for next update
        self._seed_queue.append((word, float(intensity)))

    # -------------------
    # Internal helpers
    # -------------------
    def _palette_from_emotion(self, dominant:str, valence:float, arousal:float) -> Tuple[int,int,int]:
        # choose base color: prefer mapping if present else neutral
        base = PALETTES.get(dominant, PALETTES["neutral"])
        # modulate brightness with valence
        m = 1.0 + (valence * 0.25)
        col = tuple(int(clamp(c * m, 0, 255)) for c in base)
        return col

    def _flow_field(self, x:float, y:float) -> Tuple[float,float]:
        """Produces a small guiding flow for organisms based on voice + fluid velocities."""
        # sample nearby fluid velocity
        fx = 0.0; fy = 0.0
        try:
            rx = int((x / self.W) * self.fluid.res_w); ry = int((y / self.H) * self.fluid.res_h)
            rx = clamp(rx, 0, self.fluid.res_w-1); ry = clamp(ry, 0, self.fluid.res_h-1)
            fx = float(self.fluid.velocity[ry, rx, 0])
            fy = float(self.fluid.velocity[ry, rx, 1])
        except Exception:
            pass
        # add weak global bias from voice (if engine provides)
        if self.engine and hasattr(self.engine, "voice_last") and self.engine.voice_last:
            v = self.engine.voice_last
            # voice arousal skews vertical, valence skews horizontal
            fx += (v.get("valence", 0.0)) * 0.02
            fy -= (v.get("arousal", 0.0)) * 0.02
        return fx, fy

    def _process_seeds(self):
        """Turn queued words into organisms & ink splats."""
        while self._seed_queue:
            word, intensity = self._seed_queue.pop(0)
            # pick a seed position: center-ish with jitter
            sx = self.W // 2 + random.randint(-120, 120)
            sy = int(self.H * 0.55) + random.randint(-80, 80)

            # derive emotion snapshot for color/growth
            if self.engine and hasattr(self.engine, "latest_frame") and self.engine.latest_frame:
                f = self.engine.latest_frame
                dominant = getattr(f, "dominant", "neutral")
                val = float(getattr(f, "valence", 0.0))
                aro = float(getattr(f, "arousal", 0.0))
            else:
                # fallback demo values
                dominant = random.choice(list(PALETTES.keys()))
                val = random.uniform(-0.6, 0.6)
                aro = random.uniform(0.3, 1.0)

            color = self._palette_from_emotion(dominant, val, aro)
            # create organism (dominant)
            org = Organism(sx, sy, color, valence=val, arousal=aro)
            if len(self.organisms) < self.config["max_organisms"]:
                self.organisms.append(org)

            # splat dye in fluid
            r = int(30 + 80 * intensity * (0.5 + aro))
            self.fluid.splat_dye(sx, sy, color, strength=0.6*intensity, radius_px=r)
            # add small force
            fx = random.uniform(-1,1) * (0.5 + aro)
            fy = -0.5 * (1.0 + aro)
            self.fluid.add_force(sx, sy, fx, fy, radius_px=r*2)

            # add particles
            for _ in range(int(4 + aro*8)):
                self.particles.append({
                    "x": sx + random.uniform(-r,r),
                    "y": sy + random.uniform(-r,r),
                    "vx": random.uniform(-0.4,0.4),
                    "vy": random.uniform(-1.2,-0.2),
                    "life": random.uniform(1.0, 4.0),
                    "size": random.uniform(2.0, 6.0),
                    "col": color
                })

    # -------------------
    # Update loop
    # -------------------
    def update(self, dt:float):
        # Demo mode: auto-seed periodically
        if not self.engine or not hasattr(self.engine, "latest_frame"):
            self.demo_t += dt
            if self.demo_t - self.last_auto_seed > self.config["auto_seed_interval"]:
                self.seed_word(random.choice(["demo", "test", "flow"]), random.random() * 1.2)
                self.last_auto_seed = self.demo_t
        
        # seed queue → create organisms or dye
        self._process_seeds()

        # update fluid guided by voice amplitude
        if self.engine and hasattr(self.engine, "voice_last") and self.engine.voice_last:
            v = self.engine.voice_last
            amp = clamp(v.get("arousal", 0.0), 0.0, 1.0)
            cx = int(self.W*0.5 + random.uniform(-120,120) * amp)
            cy = int(self.H*0.45 + random.uniform(-80,80) * amp)
            fx = random.uniform(-1,1) * (0.4 + amp)
            fy = random.uniform(-0.5, -0.1) * (0.4 + amp)
            self.fluid.add_force(cx, cy, fx, fy, radius_px=int(30+70*amp))
            col = self._palette_from_emotion("neutral", v.get("valence",0.0), v.get("arousal",0.0))
            self.fluid.splat_dye(cx, cy, col, strength=0.2* (1.0+amp), radius_px=int(10+60*amp))

        # step fluid (cheap)
        self.fluid.step(dt)

        # update organisms
        for org in list(self.organisms):
            org.update(dt * self.config["organism_growth_rate"], self._flow_field)
            if org.is_dead():
                try: self.organisms.remove(org)
                except: pass

        # particles
        for p in list(self.particles):
            p["x"] += p["vx"] * dt * 60
            p["y"] += p["vy"] * dt * 60
            p["vy"] += 0.02 * dt * 60  # slight gravity
            p["life"] -= dt
            p["size"] *= (0.995)
            if p["life"] <= 0 or p["y"] > self.H + 20:
                self.particles.remove(p)
        # keep particle cap
        if len(self.particles) > self.config["particle_limit"]:
            self.particles = self.particles[-self.config["particle_limit"]:]

    # -------------------
    # Rendering
    # -------------------
    def render(self):
        # base darkness
        base = int(35)
        self.screen.fill((base, base, base))

        # fluid background (30% contribution visually)
        fluid_surf = self.fluid.render_surface()
        self.screen.blit(fluid_surf, (0,0), special_flags=pygame.BLEND_MULT)

        # organisms (70% emphasis) – draw with additive glows
        organ_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for org in self.organisms:
            org.draw(organ_surf, opacity=self.config["organism_opacity"])
        self.screen.blit(organ_surf, (0,0), special_flags=pygame.BLEND_ADD)

        # particles
        for p in self.particles:
            s = int(max(1, p["size"]))
            surf = pygame.Surface((s*3, s*3), pygame.SRCALPHA)
            col = p["col"]
            alpha = int(200 * clamp(p["life"], 0.0, 1.0))
            pygame.draw.circle(surf, (*col, alpha), (s, s), s)
            self.screen.blit(surf, (int(p["x"]-s), int(p["y"]-s)), special_flags=pygame.BLEND_ADD)

        # UI overlay: show counts and optional emotion readout
        info_surf = pygame.Surface((500, 100), pygame.SRCALPHA)
        font = pygame.font.SysFont("Arial", 16)
        
        txt = f"Organisms: {len(self.organisms)}  Particles: {len(self.particles)}  Fluid: {self.fluid.res_w}x{self.fluid.res_h}"
        info = font.render(txt, True, (220,220,220))
        info_surf.blit(info, (8,8))
        
        # emotion readout
        emo_txt = ""
        if self.engine and hasattr(self.engine, "latest_frame") and self.engine.latest_frame:
            f = self.engine.latest_frame
            emo_txt = f"{f.dominant}  V:{f.valence:+.2f} A:{f.arousal:+.2f}"
        elif self.engine and hasattr(self.engine, "voice_last") and self.engine.voice_last:
            v = self.engine.voice_last
            emo_txt = f"VOICE V:{v.get('valence',0.0):+.2f} A:{v.get('arousal',0.0):+.2f}"
        else:
            emo_txt = "DEMO MODE - Press S to spawn organisms"
            
        if emo_txt:
            info2 = font.render(emo_txt, True, (200,200,255))
            info_surf.blit(info2, (8,32))
            
        # FPS display
        fps_txt = f"FPS: {int(self.clock.get_fps())}"
        fps_render = font.render(fps_txt, True, (150, 255, 150))
        info_surf.blit(fps_render, (8, 56))
        
        self.screen.blit(info_surf, (8,8))

    # -------------------
    # Run loop (call in thread or main)
    # -------------------
    def run(self):
        print("\n" + "="*70)
        print("  Reflekt Visual Engine Starting")
        print("="*70)
        print("  Controls:")
        print("    S - Manually spawn organism")
        print("    Q/ESC - Quit")
        print("    Space - Toggle pause")
        print("="*70 + "\n")
        
        self.running = True
        last = time.time()
        paused = False
        
        # Seed initial organisms for immediate visual feedback
        for _ in range(3):
            self.seed_word("init", random.uniform(0.6, 1.0))
        
        try:
            while self.running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                            self.running = False
                        elif ev.key == pygame.K_s:
                            # quick manual seed
                            self.seed_word("manual", 1.0)
                            print("→ Manual organism spawned")
                        elif ev.key == pygame.K_SPACE:
                            paused = not paused
                            print(f"→ {'Paused' if paused else 'Resumed'}")
                
                now = time.time()
                dt = min(0.1, now - last)
                last = now

                if not paused:
                    # update and render
                    self.update(dt)
                
                self.render()
                pygame.display.flip()
                self.clock.tick(self.config["fps"])
                
        except KeyboardInterrupt:
            print("\n→ Interrupted by user")
        finally:
            self.cleanup()

    def stop(self):
        self.running = False

    def cleanup(self):
        pygame.quit()
        print("✓ Visual engine stopped, pygame quit.")


# Aliases for compatibility
ReflektVisualHybrid = ReflektVisualEngine
ConfessionBoothVisualEngine = ReflektVisualEngine


# -----------------------
# Standalone demo
# -----------------------
def main():
    print("\n" + "="*70)
    print("  Reflekt Visual Engine - Standalone Test")
    print("="*70)
    print("  Running in DEMO mode (no emotion engine)")
    print("  Organisms will spawn automatically")
    print("="*70 + "\n")
    
    vis = ReflektVisualEngine(engine=None, config=CONFIG)
    try:
        vis.run()
    except Exception as e:
        print("Error in visual engine:", e)
        import traceback
        traceback.print_exc()
        vis.cleanup()

if __name__ == "__main__":
    main()