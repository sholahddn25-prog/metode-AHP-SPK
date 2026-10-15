/* ==========================================================================
   cursor3d.js — Interactive 3D cursor for AutoAHP
   Three.js particle orbs + trailing glow that follow the mouse with smooth
   spring (lerp) interpolation. Marvel Red #EA1D22 · Web Blue #2E7DF7 (Spider-Man).

   Behaviour:
     • A core "fluid orb" (icosahedron wireframe + additive glow sprite).
     • A ring of orbiting satellite particles.
     • A fading trail sampled from recent cursor history.
     • Magnetic hover: when the cursor is over [data-magnetic] elements the        orbs swell and shift toward web blue, with a subtle attraction pull.

   Public API:
     Cursor3D.start() / .stop() / .setEnabled(bool) / .setReducedMotion(bool)
   ========================================================================== */

(function () {
  "use strict";

  const SAPPHIRE = 0xea1d22; // Marvel red (primary)
  const EMERALD = 0x2e7df7;  // web blue (secondary)
  const TRAIL_LENGTH = 42;

  const state = {
    enabled: true,
    reducedMotion:
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    magnetic: 0, // 0..1 magnetic hover intensity
    target: { x: 0, y: 0 }, // raw cursor position (px)
    smooth: { x: 0, y: 0 }, // interpolated position (px)
    velocity: 0,
  };

  let renderer, scene, camera, core, glowSprite, satellites, trailPoints;
  let trailPositions, trailColors;
  let clock, rafId = null;
  let initialized = false;
  const trail = []; // recent cursor samples {x, y}

  // ----------------------------------------------------------------------- //
  // Setup
  // ----------------------------------------------------------------------- //

  function makeGlowTexture() {
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d");
    const gradient = ctx.createRadialGradient(
      size / 2, size / 2, 0,
      size / 2, size / 2, size / 2
    );
    gradient.addColorStop(0, "rgba(255,255,255,0.95)");
    gradient.addColorStop(0.25, "rgba(255,120,120,0.55)");
    gradient.addColorStop(0.6, "rgba(220,40,40,0.18)");
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
  }

  function init() {
    if (initialized || typeof THREE === "undefined") return;
    const canvas = document.getElementById("cursor-canvas");
    if (!canvas) return;

    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(
      60, window.innerWidth / window.innerHeight, 0.1, 100
    );
    camera.position.z = 6;

    // --- Core orb: glowing sprite behind a wireframe icosahedron ---------
    const spriteMaterial = new THREE.SpriteMaterial({
      map: makeGlowTexture(),
      color: SAPPHIRE,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    glowSprite = new THREE.Sprite(spriteMaterial);
    glowSprite.scale.set(2.4, 2.4, 1);
    scene.add(glowSprite);

    const coreGeometry = new THREE.IcosahedronGeometry(0.34, 1);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0xff5a5f, // suit-red wireframe
      wireframe: true,
      transparent: true,
      opacity: 0.75,
    });
    core = new THREE.Mesh(coreGeometry, coreMaterial);
    scene.add(core);

    // --- Orbiting satellite particles ------------------------------------
    const satCount = 70;
    const satGeometry = new THREE.BufferGeometry();
    const satPositions = new Float32Array(satCount * 3);
    const satPhase = new Float32Array(satCount);
    for (let i = 0; i < satCount; i++) {
      satPhase[i] = Math.random() * Math.PI * 2;
      satPositions[i * 3] = 0;
      satPositions[i * 3 + 1] = 0;
      satPositions[i * 3 + 2] = 0;
    }
    satGeometry.setAttribute("position", new THREE.BufferAttribute(satPositions, 3));
    const satMaterial = new THREE.PointsMaterial({
      size: 0.07,
      color: EMERALD,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    satellites = new THREE.Points(satGeometry, satMaterial);
    satellites.userData.phase = satPhase;
    scene.add(satellites);

    // --- Trail (fading polyline of recent cursor samples) ----------------
    const trailGeometry = new THREE.BufferGeometry();
    trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    trailColors = new Float32Array(TRAIL_LENGTH * 3);
    trailGeometry.setAttribute(
      "position", new THREE.BufferAttribute(trailPositions, 3)
    );
    trailGeometry.setAttribute(
      "color", new THREE.BufferAttribute(trailColors, 3)
    );
    const trailMaterial = new THREE.PointsMaterial({
      size: 0.16,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      map: makeGlowTexture(),
    });
    trailPoints = new THREE.Points(trailGeometry, trailMaterial);
    scene.add(trailPoints);

    // --- Events -----------------------------------------------------------
    state.target.x = state.smooth.x = window.innerWidth / 2;
    state.target.y = state.smooth.y = window.innerHeight / 2;

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("resize", onResize, { passive: true });
    document.addEventListener("mouseleave", () => {
      state.magnetic = 0;
    });
    document.addEventListener(
      "mouseover",
      (event) => {
        const hit = event.target && event.target.closest
          ? event.target.closest("[data-magnetic]")
          : null;
        state.magneticTarget = hit ? 1 : 0;
      },
      { passive: true }
    );
    state.magneticTarget = 0;

    clock = new THREE.Clock();
    initialized = true;
    loop();
  }

  // ----------------------------------------------------------------------- //
  // Helpers
  // ----------------------------------------------------------------------- //

  function pxToWorld(x, y) {
    // Map screen px to the z=0 plane of the perspective camera.
    const vector = new THREE.Vector3(
      (x / window.innerWidth) * 2 - 1,
      -(y / window.innerHeight) * 2 + 1,
      0.5
    );
    vector.unproject(camera);
    const direction = vector.sub(camera.position).normalize();
    const distance = -camera.position.z / direction.z;
    return camera.position.clone().add(direction.multiplyScalar(distance));
  }

  function onMouseMove(event) {
    state.target.x = event.clientX;
    state.target.y = event.clientY;
  }

  function onResize() {
    if (!renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  const lerp = (a, b, t) => a + (b - a) * t;

  const colorA = new THREE.Color(SAPPHIRE);
  const colorB = new THREE.Color(EMERALD);
  const tmpColor = new THREE.Color();

  // ----------------------------------------------------------------------- //
  // Render loop
  // ----------------------------------------------------------------------- //

  function loop() {
    rafId = requestAnimationFrame(loop);
    if (!state.enabled) return;

    const t = clock.getElapsedTime();

    // Spring-lerp the smoothed cursor position toward the real cursor.
    const spring = state.reducedMotion ? 1 : 0.12;
    const prevX = state.smooth.x;
    const prevY = state.smooth.y;
    state.smooth.x = lerp(state.smooth.x, state.target.x, spring);
    state.smooth.y = lerp(state.smooth.y, state.target.y, spring);
    state.velocity = lerp(
      state.velocity,
      Math.hypot(state.smooth.x - prevX, state.smooth.y - prevY),
      0.2
    );

    // Magnetic intensity eases toward its target.
    state.magnetic = lerp(state.magnetic, state.magneticTarget || 0, 0.15);

    const world = pxToWorld(state.smooth.x, state.smooth.y);
    const swell = 1 + state.magnetic * 0.9 + Math.min(state.velocity / 60, 0.35);

    core.position.set(world.x, world.y, 0);
    core.rotation.x = t * 0.6;
    core.rotation.y = t * 0.8;
    core.scale.setScalar(swell);

    glowSprite.position.set(world.x, world.y, -0.2);
    const pulse = 1 + Math.sin(t * 2.2) * 0.08;
    glowSprite.scale.set(2.4 * swell * pulse, 2.4 * swell * pulse, 1);
    tmpColor.copy(colorA).lerp(colorB, state.magnetic);
    glowSprite.material.color.copy(tmpColor);

    // Satellites orbit the core on a slowly rotating tilted ring.
    const phases = satellites.userData.phase;
    const positions = satellites.geometry.attributes.position.array;
    const radius = 0.85 * swell;
    for (let i = 0; i < phases.length; i++) {
      const angle = phases[i] + t * (0.8 + (i % 5) * 0.18);
      const r = radius * (0.6 + 0.4 * Math.sin(phases[i] * 3 + t));
      positions[i * 3] = world.x + Math.cos(angle) * r;
      positions[i * 3 + 1] = world.y + Math.sin(angle) * r * 0.7;
      positions[i * 3 + 2] = Math.sin(angle * 2) * 0.3;
    }
    satellites.geometry.attributes.position.needsUpdate = true;
    satellites.rotation.z = t * 0.15;
    satellites.material.color.copy(tmpColor);

    // Trail: record samples of the smoothed position, oldest fades out.
    trail.push({ x: state.smooth.x, y: state.smooth.y });
    while (trail.length > TRAIL_LENGTH) trail.shift();

    for (let i = 0; i < TRAIL_LENGTH; i++) {
      const sample = trail[Math.min(i, trail.length - 1)];
      const w = pxToWorld(sample ? sample.x : -9999, sample ? sample.y : -9999);
      trailPositions[i * 3] = w.x;
      trailPositions[i * 3 + 1] = w.y;
      trailPositions[i * 3 + 2] = -0.1;

      const age = i / (TRAIL_LENGTH - 1); // 0 oldest .. 1 newest
      tmpColor.copy(colorA).lerp(colorB, age * (0.4 + state.magnetic * 0.6));
      trailColors[i * 3] = tmpColor.r * age;
      trailColors[i * 3 + 1] = tmpColor.g * age;
      trailColors[i * 3 + 2] = tmpColor.b * age;
    }
    trailPoints.geometry.attributes.position.needsUpdate = true;
    trailPoints.geometry.attributes.color.needsUpdate = true;

    renderer.render(scene, camera);
  }

  // ----------------------------------------------------------------------- //
  // Pause when the tab is hidden (saves GPU while backgrounded).
  // ----------------------------------------------------------------------- //

  document.addEventListener("visibilitychange", () => {
    if (!initialized) return;
    if (document.hidden) {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    } else if (!rafId && state.enabled) {
      clock.getDelta();
      loop();
    }
  });

  // ----------------------------------------------------------------------- //
  // Public API
  // ----------------------------------------------------------------------- //

  window.Cursor3D = {
    start() {
      if (!initialized) init();
      state.enabled = true;
      document.body.classList.remove("no-cursor-3d");
      if (!rafId && initialized) loop();
    },
    stop() {
      state.enabled = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    },
    setEnabled(on) {
      if (on) this.start();
      else {
        this.stop();
        document.body.classList.add("no-cursor-3d");
      }
    },
    setReducedMotion(on) {
      state.reducedMotion = !!on;
    },
    get enabled() {
      return state.enabled;
    },
  };
})();
