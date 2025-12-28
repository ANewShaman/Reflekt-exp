// sketch.js – "The Liquid Soul"
// A living, breathing organism that morphs based on emotion

let particles = [];
let flowField = [];
let cols, rows;
let scl = 20; 
let zOff = 0; 

// STATE
let currentEmo = "neutral";
let targetColor, currentColor;
let volumeIntensity = 0;
let orbRadius; 
let isConnected = false;
let memorySnapshots = []; // Array to hold multiple snapshots
let emotionCycleTimer = 0; // Timer for emotion cycles
let lastSnapshotTime = 0; // Track when last snapshot was taken

// END SESSION STATE
let endingSession = false;
let endStartTime = 0;
const END_DURATION = 10000; // 10 seconds (Waits longer before blank screen)

// BUFFERS
let feedbackLayer;
let cracksLayer;

// CONFIG
const PARTICLE_COUNT = 1900;
const FLOW_STRENGTH = 0.8;
const ORB_SIZE_RATIO = 0.38;
const BLOOM_CYCLE_DURATION = 180; // ~3 seconds at 60fps
const RAIN_CYCLE_DURATION = 240; // ~4 seconds at 60fps
const SNAPSHOT_INTERVAL = 6000; // 6 seconds in milliseconds
const SNAPSHOT_PARTICLE_RATIO = 0.3; // 30% of particles

// EMOTION MATERIALS - Each has unique visual properties
const MATERIALS = {
  happy: { c: [255, 220, 80], c2: [255, 100, 200], mode: 'bloom', type: 'glow', expansion: 1.4 },
  surprise: { c: [255, 150, 200], c2: [255, 220, 100], mode: 'bloom', type: 'glow', expansion: 1.6 },
  sad: { c: [60, 120, 255], c2: [40, 80, 180], mode: 'rain', type: 'water', weight: 2.0 },
  neutral: { c: [180, 200, 230], c2: [150, 180, 220], mode: 'drift', type: 'smoke', flow: 0.6 },
  angry: { c: [180, 0, 0], c2: [255, 0, 0], mode: 'spike', type: 'metal', chaos: 2.5 },
  fear: { c: [160, 60, 255], c2: [120, 40, 200], mode: 'glitch', type: 'static', jitter: 3.0 },
  disgust: { c: [120, 255, 80], c2: [80, 200, 60], mode: 'glitch', type: 'static', jitter: 2.5 }
};

function setup() {
  createCanvas(windowWidth, windowHeight);
  
  feedbackLayer = createGraphics(width, height);
  cracksLayer = createGraphics(width, height);
  generateCracks(cracksLayer);
  
  background(0);
  
  cols = floor(width / scl);
  rows = floor(height / scl);
  flowField = new Array(cols * rows);
  orbRadius = height * ORB_SIZE_RATIO;

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles[i] = new Particle();
  }
  
  currentColor = color(180, 200, 230);
  lastSnapshotTime = millis();
  
  console.log("The Liquid Soul is awakening...");
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  feedbackLayer.resizeCanvas(width, height);
  cracksLayer.resizeCanvas(width, height);
  generateCracks(cracksLayer);
  
  orbRadius = height * ORB_SIZE_RATIO;
  cols = floor(width / scl);
  rows = floor(height / scl);
  flowField = new Array(cols * rows);
}

function keyPressed() {
  if (key === 'm' || key === 'M') takeSnapshot();
  if (key === 'c' || key === 'C') memorySnapshots = [];
  if (key === 'e' || key === 'E') startEndSession();
}

function startEndSession() {
  if (!endingSession) {
    endingSession = true;
    endStartTime = millis();
    console.log("Ending session... The soul fades away.");
  }
}

function takeSnapshot() {
  let snap = get();
  snap.resize(width / 3, height / 3); 
  memorySnapshots.push(snap);
  console.log("Memory captured (" + memorySnapshots.length + " total)");
  lastSnapshotTime = millis();
}

function draw() {
  // === END SESSION ANIMATION ===
  if (endingSession) {
    let elapsed = millis() - endStartTime;
    let progress = elapsed / END_DURATION; 
    
    if (progress >= 1.0) {
      // Final black screen - stop drawing
      background(0);
      
      // Display end message (Styled like index.html)
      push();
      fill(255);
      noStroke();
      textSize(24);
      textAlign(CENTER, CENTER);
      textFont('Courier New'); // Match Flashcard font
      
      text("SESSION TERMINATED", width/2, height/2);
      
      textSize(12);
      fill(150); // Grey subtitle
      text("BIOMETRIC SEQUENCE COMPLETE", width/2, height/2 + 30);
      pop();
      
      return;
    }
    
    // === MEMORY LAYER ===
    if (memorySnapshots.length > 0) {
      push();
      noSmooth();
      for (let snap of memorySnapshots) {
        image(snap, 0, 0, width, height);
      }
      fill(0, 10 + progress * 30);
      rect(0, 0, width, height);
      pop();
    } else {
      background(0);
    }
    
    // === GHOST TRAIL LAYER ===
    push();
    tint(255, 240 * (1 - progress));
    imageMode(CENTER);
    image(feedbackLayer, width/2, height/2, width * 1.003, height * 1.003);
    pop();
    
    // === PHYSICS ===
    calculateFlowField();
    
    // === RENDER PARTICLES ===
    let particleSubsetCount = floor(PARTICLE_COUNT * SNAPSHOT_PARTICLE_RATIO);
    let skipInterval = floor(PARTICLE_COUNT / particleSubsetCount);
    
    for (let i = 0; i < particles.length; i++) {
      let p = particles[i];
      p.follow(flowField);
      p.update();
      p.checkOrbBounds();
      
      if (progress > 0.3) {
        let suckProgress = (progress - 0.3) / 0.7; 
        let toCenter = createVector(width/2 - p.pos.x, height/2 - p.pos.y);
        toCenter.setMag(suckProgress * 8); 
        p.applyForce(toCenter);
      }
      
      let fadeProgress = min(1.0, progress / 0.3);
      p.fadeAmount = 1.0 - fadeProgress;
      
      if (i % skipInterval === 0 || skipInterval === 0) {
        p.show();
      }
    }
    
    feedbackLayer.clear();
    feedbackLayer.drawingContext.drawImage(drawingContext.canvas, 0, 0, width, height);
    
    return; 
  }
  
  // === NORMAL RENDERING ===
  
  if (memorySnapshots.length > 0) {
    push();
    noSmooth(); 
    for (let snap of memorySnapshots) {
      image(snap, 0, 0, width, height);
    }
    fill(0, 10); 
    rect(0, 0, width, height);
    pop();
  } else {
    noStroke();
    fill(0, 18);
    rect(0, 0, width, height);
  }

  push();
  tint(255, 240);
  imageMode(CENTER);
  image(feedbackLayer, width/2, height/2, width * 1.003, height * 1.003);
  pop();

  // === DATA UPDATE ===
  if (window.latestEmotion) {
    isConnected = true;
    let e = window.latestEmotion;
    let dom = e.dominant || "neutral";
    let targetVol = e.volume || 0;
    
    volumeIntensity = lerp(volumeIntensity, targetVol, 0.12);
    
    if (e.arousal > 0.95 && frameCount % 100 === 0) {
      takeSnapshot();
    }
    
    if (dom !== currentEmo) {
      if (random() < 0.7) {
        takeSnapshot();
      }
      currentEmo = dom;
      emotionCycleTimer = 0; 
      console.log(`Emotion shift: ${currentEmo.toUpperCase()}`);
    }
    
    let mat = MATERIALS[dom] || MATERIALS.neutral;
    let tc = lerpColor(
      color(mat.c[0], mat.c[1], mat.c[2]),
      color(mat.c2[0], mat.c2[1], mat.c2[2]),
      sin(frameCount * 0.03) * 0.5 + 0.5 
    );
    currentColor = lerpColor(currentColor, tc, 0.06);
  } else {
    isConnected = false;
    volumeIntensity = (sin(frameCount * 0.04) + 1) * 0.15; 
    
    let neutral = MATERIALS.neutral;
    let tc = color(neutral.c[0], neutral.c[1], neutral.c[2]);
    currentColor = lerpColor(currentColor, tc, 0.03);
  }

  emotionCycleTimer++;

  let currentTime = millis();
  if (currentTime - lastSnapshotTime >= SNAPSHOT_INTERVAL) {
    takeSnapshot();
  }

  // === PHYSICS ===
  calculateFlowField();
  
  let particleSubsetCount = floor(PARTICLE_COUNT * SNAPSHOT_PARTICLE_RATIO);
  let skipInterval = floor(PARTICLE_COUNT / particleSubsetCount);
  
  for (let i = 0; i < particles.length; i++) {
    let p = particles[i];
    p.follow(flowField);
    p.update();
    p.checkOrbBounds();
    
    if (i % skipInterval === 0 || skipInterval === 0) {
      p.show();
    }
  }

  if (currentEmo === 'fear' || currentEmo === 'disgust') {
    push();
    blendMode(OVERLAY);
    tint(255, 100 + volumeIntensity * 155);
    image(cracksLayer, 0, 0);
    blendMode(BLEND);
    pop();
  }
  
  feedbackLayer.clear();
  feedbackLayer.drawingContext.drawImage(drawingContext.canvas, 0, 0, width, height);
}

function calculateFlowField() {
  let mat = MATERIALS[currentEmo] || MATERIALS.neutral;
  let mode = mat.mode;
  let yOff = 0;
  let breathe = sin(frameCount * 0.025) * 0.15;
  
  for (let y = 0; y < rows; y++) {
    let xOff = 0;
    for (let x = 0; x < cols; x++) {
      let index = x + y * cols;
      let angle = noise(xOff, yOff, zOff) * TWO_PI * 2;
      let v = p5.Vector.fromAngle(angle);
      
      let px = x * scl;
      let py = y * scl;
      let centerX = width / 2;
      let centerY = height / 2;
      
      if (mode === 'bloom') {
        let fromCenter = createVector(px - centerX, py - centerY);
        fromCenter.normalize();
        let cycleProgress = (emotionCycleTimer % BLOOM_CYCLE_DURATION) / BLOOM_CYCLE_DURATION;
        let bloomWave = sin(cycleProgress * PI); 
        v.lerp(fromCenter, 0.4 + bloomWave * 0.6 + volumeIntensity * 0.3);
      } else if (mode === 'rain') {
        let gravity = createVector(sin(xOff * 4) * 0.3, 1.2);
        let cycleProgress = (emotionCycleTimer % RAIN_CYCLE_DURATION) / RAIN_CYCLE_DURATION;
        let rainIntensity = sin(cycleProgress * PI);
        v.lerp(gravity, 0.5 + rainIntensity * 0.4);
      } else if (mode === 'spike') {
        let fromCenter = createVector(px - centerX, py - centerY);
        let dist = fromCenter.mag();
        fromCenter.normalize();
        let turbulence = noise(xOff * 5, yOff * 5, frameCount * 0.05) * TWO_PI;
        fromCenter.rotate(turbulence);
        let explosionForce = mat.chaos + volumeIntensity * 3.5;
        if (dist < orbRadius) {
          explosionForce *= (1.0 + (orbRadius - dist) / orbRadius);
        }
        v = fromCenter.copy();
        v.setMag(explosionForce);
      } else if (mode === 'glitch') {
        v.rotate(random(-mat.jitter, mat.jitter));
        v.add(createVector(random(-1, 1), random(-1, 1)));
      } else {
        let spiral = createVector(-(py - centerY), (px - centerX));
        spiral.normalize();
        v.lerp(spiral, 0.7 + breathe);
      }
      
      let flowMag = FLOW_STRENGTH + volumeIntensity * 1.5;
      flowMag = max(flowMag, 0.3);
      if (mode !== 'spike') {
        v.setMag(flowMag);
      }
      
      flowField[index] = v;
      xOff += 0.12;
    }
    yOff += 0.12;
  }
  zOff += 0.008 + volumeIntensity * 0.03;
}

class Particle {
  constructor() {
    this.spawnInOrb();
    this.vel = createVector(0, 0);
    this.acc = createVector(0, 0);
    this.maxSpeed = random(2, 4.5);
    this.prevPos = this.pos.copy();
    this.strokeW = random(0.8, 2.5);
    this.size = random(6, 12); 
    this.life = random(0.8, 1.0);
    this.fadeAmount = 1.0; 
  }

  spawnInOrb() {
    let ang = random(TWO_PI);
    let r = random(orbRadius * 0.85); 
    this.pos = createVector(
      width/2 + cos(ang) * r, 
      height/2 + sin(ang) * r
    );
    this.prevPos = this.pos.copy();
  }

  follow(vectors) {
    let x = floor(this.pos.x / scl);
    let y = floor(this.pos.y / scl);
    x = constrain(x, 0, cols - 1);
    y = constrain(y, 0, rows - 1);
    let index = x + y * cols;
    if (vectors[index]) {
      this.applyForce(vectors[index]);
    }
  }

  applyForce(force) { 
    this.acc.add(force); 
  }

  update() {
    this.vel.add(this.acc);
    let speedMult = 1 + volumeIntensity * 0.8;
    if (currentEmo === 'angry') {
      speedMult *= 1.5;
    }
    this.vel.limit(this.maxSpeed * speedMult);
    this.pos.add(this.vel);
    this.acc.mult(0);
    this.life = lerp(this.life, 1.0, 0.02);
  }

  checkOrbBounds() {
    let d = dist(this.pos.x, this.pos.y, width/2, height/2);
    let mat = MATERIALS[currentEmo] || MATERIALS.neutral;
    let mode = mat.mode;
    
    let boundaryMult = 1.0;
    if (mode === 'bloom') {
      boundaryMult = mat.expansion || 1.4;
    } else if (mode === 'rain') {
      if (this.pos.y > height + 50) {
        this.pos.x = width/2 + random(-orbRadius * 0.8, orbRadius * 0.8);
        this.pos.y = height/2 - orbRadius;
        this.vel.y = 0; 
        this.life = random(0.8, 1.0);
        return;
      }
      boundaryMult = 1.3;
    } else if (mode === 'spike') {
      boundaryMult = 2.0;
    }
    
    let maxDist = orbRadius * boundaryMult;
    
    if (d > maxDist) {
      let centerPull = createVector(width/2 - this.pos.x, height/2 - this.pos.y);
      centerPull.normalize().mult(1.2);
      this.applyForce(centerPull);
      if (d > maxDist + 80) {
        this.spawnInOrb();
        this.life = random(0.7, 1.0);
      }
    }
  }

  show() {
    let mat = MATERIALS[currentEmo] || MATERIALS.neutral;
    let matType = mat.type;
    
    let baseAlpha = 120 + volumeIntensity * 135;
    baseAlpha *= this.life; 
    baseAlpha *= this.fadeAmount; 
    
    if (matType === 'glow') {
      noStroke();
      let glowSize = this.size * (2.5 + volumeIntensity * 2);
      fill(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.15);
      ellipse(this.pos.x, this.pos.y, glowSize);
      fill(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.5);
      ellipse(this.pos.x, this.pos.y, this.size * 1.2);
      
    } else if (matType === 'water') {
      stroke(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.8);
      strokeWeight(this.strokeW * 1.2);
      line(this.pos.x, this.pos.y, this.prevPos.x, this.prevPos.y);
      noStroke();
      fill(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.4);
      ellipse(this.pos.x, this.pos.y, this.size * 0.8);
      
    } else if (matType === 'metal') {
      stroke(red(currentColor), green(currentColor), blue(currentColor), baseAlpha);
      strokeWeight(this.strokeW * 2.0);
      let dir = p5.Vector.sub(this.pos, this.prevPos);
      dir.mult(5 + volumeIntensity * 4); 
      line(this.pos.x, this.pos.y, this.pos.x - dir.x, this.pos.y - dir.y);
      noStroke();
      fill(255, 200, 100, baseAlpha * 0.9); 
      ellipse(this.pos.x, this.pos.y, this.size * 0.8);
      if (random() < 0.4) {
        fill(255, 150, 50, baseAlpha * 0.6);
        ellipse(this.pos.x + random(-3, 3), this.pos.y + random(-3, 3), this.size * 0.4);
      }
      
    } else if (matType === 'static') {
      stroke(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.9);
      strokeWeight(1);
      let jx = random(-4, 4);
      let jy = random(-4, 4);
      line(this.pos.x + jx, this.pos.y + jy, this.prevPos.x, this.prevPos.y);
      if (random() < 0.3) {
        noStroke();
        fill(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.6);
        rect(this.pos.x + random(-3, 3), this.pos.y + random(-3, 3), 2, 2);
      }
      
    } else {
      stroke(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.7);
      strokeWeight(this.strokeW);
      line(this.pos.x, this.pos.y, this.prevPos.x, this.prevPos.y);
      noStroke();
      fill(red(currentColor), green(currentColor), blue(currentColor), baseAlpha * 0.3);
      ellipse(this.pos.x, this.pos.y, this.size);
    }
    this.prevPos.x = this.pos.x;
    this.prevPos.y = this.pos.y;
  }
}

function generateCracks(pg) {
  pg.clear();
  pg.stroke(255, 200);
  pg.strokeWeight(2);
  pg.noFill();
  let centerX = width / 2;
  let centerY = height / 2;
  for (let i = 0; i < 8; i++) {
    let angle = random(TWO_PI);
    let len = random(orbRadius * 0.6, orbRadius * 1.8);
    pg.beginShape();
    pg.vertex(centerX, centerY);
    let segments = 3;
    for (let s = 1; s <= segments; s++) {
      let t = s / segments;
      let x = centerX + cos(angle) * len * t + random(-30, 30);
      let y = centerY + sin(angle) * len * t + random(-30, 30);
      pg.vertex(x, y);
    }
    pg.endShape();
  }
  for (let i = 0; i < 12; i++) {
    let x = centerX + random(-orbRadius, orbRadius);
    let y = centerY + random(-orbRadius, orbRadius);
    let len = random(20, 60);
    let angle = random(TWO_PI);
    pg.line(x, y, x + cos(angle) * len, y + sin(angle) * len);
  }
}