const canvas = document.getElementById("board");
const context = canvas.getContext("2d");

const titleLabel = document.getElementById("titleLabel");
const appleLabel = document.getElementById("appleLabel");
const lengthLabel = document.getElementById("lengthLabel");
const trajectoryPicker = document.getElementById("trajectoryPicker");
const timeline = document.getElementById("timeline");

const TRAJECTORY_URL = new URL("../data/trajectory.json", import.meta.url);
const TRAJECTORY_INDEX_URL = new URL("../data/trajectories.json", import.meta.url);

const state = {
  trajectory: null,
  trajectoryOptions: [],
  trajectoryFile: "",
  frameIndex: 0,
  playing: true,
  accumulator: 0,
  lastTimestamp: 0,
};

function setLoadError(message) {
  state.trajectory = null;
  state.playing = false;
  state.frameIndex = 0;
  state.accumulator = 0;
  state.lastTimestamp = 0;
  titleLabel.textContent = message;
  appleLabel.textContent = "0 / 0";
  lengthLabel.textContent = "0";
  timeline.max = "0";
  timeline.value = "0";
  render();
}

function normalizeTrajectoryOption(option, index) {
  if (!option || typeof option !== "object") {
    throw new Error(`Trajectory option ${index} is invalid.`);
  }

  const file = String(option.file || "").trim();
  if (!file) {
    throw new Error(`Trajectory option ${index} is missing a file.`);
  }

  return {
    label: String(option.label || file),
    file,
    group: String(option.group || ""),
  };
}

function populateTrajectoryPicker() {
  trajectoryPicker.replaceChildren();

  const groups = new Map();

  state.trajectoryOptions.forEach((option) => {
    const element = document.createElement("option");
    element.value = option.file;
    element.textContent = option.label;

    if (!option.group) {
      trajectoryPicker.append(element);
      return;
    }

    if (!groups.has(option.group)) {
      const group = document.createElement("optgroup");
      group.label = option.group;
      groups.set(option.group, group);
      trajectoryPicker.append(group);
    }

    groups.get(option.group).append(element);
  });

  trajectoryPicker.disabled = state.trajectoryOptions.length <= 1;
  if (state.trajectoryFile) {
    trajectoryPicker.value = state.trajectoryFile;
  }
}

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

function updateUi() {
  if (!state.trajectory) {
    return;
  }

  const { frames, apples } = state.trajectory;
  const currentFrame = frames[state.frameIndex];
  const eatenCount = currentFrame.eaten.size;

  titleLabel.textContent = state.trajectory.title;
  appleLabel.textContent = `${eatenCount} / ${apples.length}`;
  lengthLabel.textContent = String(state.frameIndex);
  timeline.value = String(state.frameIndex);
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

    while (state.accumulator >= state.trajectory.frameDurationMs) {
      state.accumulator -= state.trajectory.frameDurationMs;
      advanceFrame();
    }
  }

  requestAnimationFrame(animationLoop);
}

async function loadTrajectory() {
  const response = await fetch(new URL(`../data/${state.trajectoryFile}`, import.meta.url));
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

async function loadTrajectoryOptions() {
  const response = await fetch(TRAJECTORY_INDEX_URL);
  if (response.ok) {
    const payload = await response.json();
    if (!Array.isArray(payload) || payload.length === 0) {
      throw new Error("Trajectory index must contain at least one option.");
    }
    state.trajectoryOptions = payload.map(normalizeTrajectoryOption);
  } else if (response.status === 404) {
    state.trajectoryOptions = [{
      label: "A*",
      file: TRAJECTORY_URL.pathname.split("/").pop(),
      group: "Search",
    }];
  } else {
    throw new Error(`Unable to load trajectory index: ${response.status}`);
  }

  state.trajectoryFile = state.trajectoryOptions[0].file;
  populateTrajectoryPicker();
}

async function selectTrajectory(file) {
  state.trajectoryFile = file;
  trajectoryPicker.value = file;
  titleLabel.textContent = "Loading...";
  appleLabel.textContent = "0 / 0";
  lengthLabel.textContent = "0";
  await loadTrajectory();
}

timeline.addEventListener("input", (event) => {
  state.frameIndex = Number(event.target.value);
  state.accumulator = 0;
  updateUi();
  render();
});

trajectoryPicker.addEventListener("change", async (event) => {
  try {
    await selectTrajectory(event.target.value);
  } catch (error) {
    setLoadError("Load failed");
    console.error(error);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || !state.trajectory) {
    return;
  }

  event.preventDefault();
  state.playing = !state.playing;
});

window.addEventListener("resize", resizeCanvas);

requestAnimationFrame(animationLoop);

loadTrajectoryOptions()
  .then(() => selectTrajectory(state.trajectoryFile))
  .catch((error) => {
    setLoadError("Load failed");
    console.error(error);
  });
