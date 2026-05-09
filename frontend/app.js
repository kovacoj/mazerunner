const canvas = document.getElementById("board");
const context = canvas.getContext("2d");

const titleLabel = document.getElementById("titleLabel");
const frameLabel = document.getElementById("frameLabel");
const appleLabel = document.getElementById("appleLabel");
const speedLabel = document.getElementById("speedLabel");
const statusLabel = document.getElementById("statusLabel");
const toggleButton = document.getElementById("toggleButton");
const timeline = document.getElementById("timeline");
const speedControl = document.getElementById("speedControl");

const TRAJECTORY_URL = new URL("../data/trajectory.json", import.meta.url);

const state = {
  trajectory: null,
  frameIndex: 0,
  playing: true,
  speed: 1,
  accumulator: 0,
  lastTimestamp: 0,
};

function setStatus(message) {
  statusLabel.textContent = message;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizePoint(point) {
  return {
    x: Number(point.x),
    y: Number(point.y),
  };
}

function normalizeTrajectory(payload) {
  if (!Array.isArray(payload.frames) || payload.frames.length === 0) {
    throw new Error("Trajectory requires at least one frame.");
  }

  const gridSize = Number(payload.gridSize || 20);
  const frameDurationMs = Number(payload.frameDurationMs || 160);
  const apples = Array.isArray(payload.apples) ? payload.apples.map(normalizePoint) : [];

  const frames = payload.frames.map((frame, index) => {
    if (!Array.isArray(frame.snake) || frame.snake.length === 0) {
      throw new Error(`Frame ${index} is missing snake segments.`);
    }

    return {
      snake: frame.snake.map(normalizePoint),
      eaten: new Set(Array.isArray(frame.eaten) ? frame.eaten : []),
    };
  });

  return {
    title: payload.title || "Snake trajectory",
    gridSize,
    frameDurationMs,
    apples,
    frames,
  };
}

function resizeCanvas() {
  const size = canvas.clientWidth;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  render();
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);

  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function colorForSegment(index, segmentCount) {
  const ratio = segmentCount <= 1 ? 1 : index / (segmentCount - 1);
  const lightness = 62 - ratio * 28;
  return `hsl(136 86% ${lightness}%)`;
}

function updateUi() {
  if (!state.trajectory) {
    return;
  }

  const { frames, apples } = state.trajectory;
  const currentFrame = frames[state.frameIndex];
  const eatenCount = currentFrame.eaten.size;

  titleLabel.textContent = state.trajectory.title;
  frameLabel.textContent = `${state.frameIndex + 1} / ${frames.length}`;
  appleLabel.textContent = `${eatenCount} / ${apples.length}`;
  speedLabel.textContent = `${state.speed.toFixed(2)}x`;
  timeline.value = String(state.frameIndex);
  toggleButton.textContent = state.playing ? "Pause" : state.frameIndex === frames.length - 1 ? "Replay" : "Play";
}

function getBoardMetrics() {
  const size = canvas.clientWidth;
  const padding = Math.max(12, size * 0.03);
  const boardSize = size - padding * 2;
  const gridSize = state.trajectory?.gridSize || 20;
  const cell = boardSize / gridSize;

  return { size, padding, boardSize, cell, gridSize };
}

function centerOf(point, metrics) {
  return {
    x: metrics.padding + (point.x + 0.5) * metrics.cell,
    y: metrics.padding + (point.y + 0.5) * metrics.cell,
  };
}

function drawGrid(metrics) {
  context.fillStyle = "#0b1118";
  roundedRect(context, metrics.padding, metrics.padding, metrics.boardSize, metrics.boardSize, 22);
  context.fill();

  context.strokeStyle = "rgba(255, 255, 255, 0.06)";
  context.lineWidth = 1;

  for (let index = 0; index <= metrics.gridSize; index += 1) {
    const offset = metrics.padding + index * metrics.cell;

    context.beginPath();
    context.moveTo(metrics.padding, offset);
    context.lineTo(metrics.padding + metrics.boardSize, offset);
    context.stroke();

    context.beginPath();
    context.moveTo(offset, metrics.padding);
    context.lineTo(offset, metrics.padding + metrics.boardSize);
    context.stroke();
  }
}

function drawTrail(metrics) {
  const frames = state.trajectory.frames.slice(0, state.frameIndex + 1);
  if (frames.length < 2) {
    return;
  }

  context.save();
  context.strokeStyle = "rgba(101, 243, 140, 0.22)";
  context.lineWidth = Math.max(4, metrics.cell * 0.18);
  context.lineCap = "round";
  context.lineJoin = "round";

  context.beginPath();
  frames.forEach((frame, index) => {
    const point = centerOf(frame.snake[0], metrics);
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  context.stroke();
  context.restore();
}

function drawApples(metrics, currentFrame) {
  state.trajectory.apples.forEach((apple, index) => {
    const point = centerOf(apple, metrics);
    const eaten = currentFrame.eaten.has(index);
    const radius = metrics.cell * 0.22;

    context.save();
    context.fillStyle = eaten ? "rgba(255, 107, 107, 0.16)" : "#ff6b6b";
    context.shadowBlur = eaten ? 0 : 18;
    context.shadowColor = "rgba(255, 107, 107, 0.35)";
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();

    if (eaten) {
      context.lineWidth = 2;
      context.strokeStyle = "rgba(255, 107, 107, 0.35)";
      context.beginPath();
      context.arc(point.x, point.y, radius + 4, 0, Math.PI * 2);
      context.stroke();
    }

    context.restore();
  });
}

function drawSnake(metrics, currentFrame) {
  const inset = metrics.cell * 0.12;
  const segmentSize = metrics.cell - inset * 2;

  currentFrame.snake
    .slice()
    .reverse()
    .forEach((segment, reverseIndex, segments) => {
      const index = segments.length - reverseIndex - 1;
      const x = metrics.padding + segment.x * metrics.cell + inset;
      const y = metrics.padding + segment.y * metrics.cell + inset;

      context.save();
      context.fillStyle = colorForSegment(index, segments.length);
      context.shadowBlur = index === 0 ? 22 : 0;
      context.shadowColor = "rgba(101, 243, 140, 0.34)";
      roundedRect(context, x, y, segmentSize, segmentSize, segmentSize * 0.32);
      context.fill();
      context.restore();
    });
}

function render() {
  const width = canvas.clientWidth;
  if (!width) {
    return;
  }

  context.clearRect(0, 0, width, width);

  if (!state.trajectory) {
    return;
  }

  const metrics = getBoardMetrics();
  const currentFrame = state.trajectory.frames[state.frameIndex];

  drawGrid(metrics);
  drawTrail(metrics);
  drawApples(metrics, currentFrame);
  drawSnake(metrics, currentFrame);
}

function advanceFrame() {
  if (!state.trajectory) {
    return;
  }

  if (state.frameIndex >= state.trajectory.frames.length - 1) {
    state.playing = false;
  } else {
    state.frameIndex += 1;
  }

  updateUi();
  render();
}

function animationLoop(timestamp) {
  if (!state.lastTimestamp) {
    state.lastTimestamp = timestamp;
  }

  const delta = timestamp - state.lastTimestamp;
  state.lastTimestamp = timestamp;

  if (state.playing && state.trajectory) {
    state.accumulator += delta * state.speed;

    while (state.accumulator >= state.trajectory.frameDurationMs && state.playing) {
      state.accumulator -= state.trajectory.frameDurationMs;
      advanceFrame();
    }
  }

  requestAnimationFrame(animationLoop);
}

async function loadTrajectory() {
  setStatus(`Loading ${TRAJECTORY_URL.pathname.split("/").pop()}...`);

  const response = await fetch(TRAJECTORY_URL);
  if (!response.ok) {
    throw new Error(`Unable to load trajectory: ${response.status}`);
  }

  const payload = await response.json();
  state.trajectory = normalizeTrajectory(payload);
  state.frameIndex = 0;
  state.accumulator = 0;
  state.lastTimestamp = 0;

  timeline.max = String(Math.max(0, state.trajectory.frames.length - 1));
  timeline.value = "0";
  updateUi();
  resizeCanvas();
  setStatus("Loaded backend trajectory from data/trajectory.json");
}

toggleButton.addEventListener("click", () => {
  if (!state.trajectory) {
    return;
  }

  if (!state.playing && state.frameIndex === state.trajectory.frames.length - 1) {
    state.frameIndex = 0;
    state.accumulator = 0;
  }

  state.playing = !state.playing;
  updateUi();
  render();
});

timeline.addEventListener("input", (event) => {
  state.frameIndex = Number(event.target.value);
  state.accumulator = 0;
  updateUi();
  render();
});

speedControl.addEventListener("input", (event) => {
  state.speed = clamp(Number(event.target.value), 0.25, 2);
  updateUi();
});

window.addEventListener("resize", resizeCanvas);

requestAnimationFrame(animationLoop);

loadTrajectory().catch((error) => {
  state.playing = false;
  setStatus(error.message);
  titleLabel.textContent = "Load failed";
  frameLabel.textContent = "0 / 0";
  appleLabel.textContent = "0 / 0";
  toggleButton.textContent = "Play";
});
