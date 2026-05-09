const canvas = document.getElementById("board");
const context = canvas.getContext("2d");

const titleLabel = document.getElementById("titleLabel");
const appleLabel = document.getElementById("appleLabel");
const lengthLabel = document.getElementById("lengthLabel");
const toggleButton = document.getElementById("toggleButton");
const timeline = document.getElementById("timeline");

const TRAJECTORY_URL = new URL("../data/trajectory.json", import.meta.url);

const state = {
  trajectory: null,
  frameIndex: 0,
  playing: true,
  accumulator: 0,
  lastTimestamp: 0,
};

function normalizePoint(point) {
  return {
    x: Number(point.x),
    y: Number(point.y),
  };
}

function mod(value, divisor) {
  return ((value % divisor) + divisor) % divisor;
}

function normalizePointForBoard(point, gridSize, wallMode) {
  if (wallMode !== "wrap") {
    return point;
  }

  return {
    x: mod(point.x, gridSize),
    y: mod(point.y, gridSize),
  };
}

function normalizeTrajectory(payload) {
  if (!Array.isArray(payload.frames) || payload.frames.length === 0) {
    throw new Error("Trajectory requires at least one frame.");
  }

  const gridSize = Number(payload.gridSize || 20);
  const frameDurationMs = Number(payload.frameDurationMs || 160);
  const wallMode = payload.wallMode === "wrap" ? "wrap" : "bounded";
  const apples = Array.isArray(payload.apples)
    ? payload.apples.map((point) => normalizePointForBoard(normalizePoint(point), gridSize, wallMode))
    : [];

  const frames = payload.frames.map((frame, index) => {
    if (!Array.isArray(frame.snake) || frame.snake.length === 0) {
      throw new Error(`Frame ${index} is missing snake segments.`);
    }

    return {
      snake: frame.snake.map((point) => normalizePointForBoard(normalizePoint(point), gridSize, wallMode)),
      eaten: new Set(Array.isArray(frame.eaten) ? frame.eaten : []),
    };
  });

  return {
    title: payload.title || "Snake trajectory",
    gridSize,
    frameDurationMs,
    wallMode,
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

function updateToggleButton() {
  if (!state.trajectory) {
    toggleButton.textContent = "\u25b6";
    toggleButton.setAttribute("aria-label", "Play replay");
    toggleButton.title = "Play replay";
    return;
  }

  if (state.playing) {
    toggleButton.textContent = "\u23f8";
    toggleButton.setAttribute("aria-label", "Pause replay");
    toggleButton.title = "Pause replay";
    return;
  }

  if (state.frameIndex === state.trajectory.frames.length - 1) {
    toggleButton.textContent = "\u21ba";
    toggleButton.setAttribute("aria-label", "Replay trajectory");
    toggleButton.title = "Replay trajectory";
    return;
  }

  toggleButton.textContent = "\u25b6";
  toggleButton.setAttribute("aria-label", "Play replay");
  toggleButton.title = "Play replay";
}

function updateUi() {
  if (!state.trajectory) {
    return;
  }

  const { frames, apples } = state.trajectory;
  const currentFrame = frames[state.frameIndex];
  const eatenCount = currentFrame.eaten.size;
  const totalLength = Math.max(frames.length - 1, 0);

  titleLabel.textContent = state.trajectory.title;
  appleLabel.textContent = `${eatenCount} / ${apples.length}`;
  lengthLabel.textContent = `${state.frameIndex} / ${totalLength}`;
  timeline.value = String(state.frameIndex);
  updateToggleButton();
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
    state.accumulator += delta;

    while (state.accumulator >= state.trajectory.frameDurationMs && state.playing) {
      state.accumulator -= state.trajectory.frameDurationMs;
      advanceFrame();
    }
  }

  requestAnimationFrame(animationLoop);
}

async function loadTrajectory() {
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

window.addEventListener("resize", resizeCanvas);

requestAnimationFrame(animationLoop);

loadTrajectory().catch((error) => {
  state.playing = false;
  titleLabel.textContent = "Load failed";
  appleLabel.textContent = "0 / 0";
  lengthLabel.textContent = "0 / 0";
  timeline.max = "0";
  timeline.value = "0";
  updateToggleButton();
  console.error(error);
});
