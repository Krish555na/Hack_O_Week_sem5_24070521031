/**
 * AuraBoard — High-Performance Studio Whiteboard Engine
 * HTML5 Canvas, Freehand Bezier Smoothing, Vector Shapes, Sticky Notes, Undo/Redo & Image Export
 */

class WhiteboardEngine {
  constructor() {
    this.canvas = document.getElementById('whiteboard-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.container = document.getElementById('canvas-container');
    this.stickyContainer = document.getElementById('sticky-notes-container');

    // Drawing state
    this.currentTool = 'brush';
    this.currentColor = '#ffffff';
    this.strokeWidth = 4;
    this.fillShape = false;
    this.isDrawing = false;
    this.startX = 0;
    this.startY = 0;
    this.snapshot = null;

    // Zoom & pan
    this.scale = 1.0;

    // History stack
    this.history = [];
    this.historyIndex = -1;
    this.maxHistory = 30;

    // Freehand points buffer
    this.currentPoints = [];

    this.initCanvasSize();
    this.bindDOM();
    this.bindCanvasEvents();
    this.bindShortcuts();
    this.saveState();
  }

  initCanvasSize() {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;

    this.ctx.scale(dpr, dpr);
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    document.getElementById('dimensions-display').textContent =
      `Canvas: ${Math.round(rect.width)} × ${Math.round(rect.height)} px`;
  }

  bindDOM() {
    // Tools
    this.toolButtons = document.querySelectorAll('[data-tool]');
    this.toolButtons.forEach(btn => {
      btn.addEventListener('click', () => this.setTool(btn.dataset.tool));
    });

    // Swatches
    this.swatches = document.querySelectorAll('.color-swatch');
    this.swatches.forEach(swatch => {
      swatch.addEventListener('click', () => {
        this.swatches.forEach(s => s.classList.remove('active'));
        swatch.classList.add('active');
        this.currentColor = swatch.dataset.color;
      });
    });

    const customColor = document.getElementById('custom-color-input');
    customColor.addEventListener('input', (e) => {
      this.currentColor = e.target.value;
      this.swatches.forEach(s => s.classList.remove('active'));
    });

    // Stroke size
    const slider = document.getElementById('stroke-slider');
    const strokeVal = document.getElementById('stroke-value');
    slider.addEventListener('input', (e) => {
      this.strokeWidth = parseInt(e.target.value, 10);
      strokeVal.textContent = `${this.strokeWidth}px`;
    });

    // Fill toggle
    const fillCheck = document.getElementById('fill-checkbox');
    fillCheck.addEventListener('change', (e) => {
      this.fillShape = e.target.checked;
    });

    // Actions
    document.getElementById('btn-undo').addEventListener('click', () => this.undo());
    document.getElementById('btn-redo').addEventListener('click', () => this.redo());
    document.getElementById('btn-clear').addEventListener('click', () => this.clearCanvas());
    document.getElementById('btn-export-png').addEventListener('click', () => this.exportPNG());
    document.getElementById('btn-save-json').addEventListener('click', () => this.saveProjectJSON());

    // Zoom
    document.getElementById('btn-zoom-in').addEventListener('click', () => this.adjustZoom(0.1));
    document.getElementById('btn-zoom-out').addEventListener('click', () => this.adjustZoom(-0.1));
    document.getElementById('btn-reset-view').addEventListener('click', () => this.resetZoom());

    // Window resize
    window.addEventListener('resize', () => {
      // Preserve canvas drawing
      const tempImg = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      this.initCanvasSize();
      this.ctx.putImageData(tempImg, 0, 0);
    });
  }

  setTool(tool) {
    this.currentTool = tool;
    this.toolButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tool === tool));
    document.getElementById('current-tool-text').textContent = `Tool: ${this.capitalize(tool)}`;

    if (tool === 'sticky') {
      this.createStickyNote(window.innerWidth / 2 - 100, window.innerHeight / 2 - 80);
      this.setTool('brush'); // Return to brush after placing sticky
    }
  }

  bindCanvasEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseup', () => this.onMouseUp());
    this.canvas.addEventListener('mouseleave', () => this.onMouseUp());

    // Touch support for tablets/hybrids
    this.canvas.addEventListener('touchstart', (e) => {
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
      e.preventDefault();
    }, { passive: false });

    this.canvas.addEventListener('touchend', () => {
      this.canvas.dispatchEvent(new MouseEvent('mouseup', {}));
    });
  }

  getPointerPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left),
      y: (e.clientY - rect.top)
    };
  }

  onMouseDown(e) {
    this.isDrawing = true;
    const pos = this.getPointerPos(e);
    this.startX = pos.x;
    this.startY = pos.y;

    // Snapshot current image before dynamic drag preview
    this.snapshot = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

    if (this.currentTool === 'brush' || this.currentTool === 'eraser') {
      this.currentPoints = [pos];
      this.ctx.beginPath();
      this.ctx.moveTo(pos.x, pos.y);
    } else if (this.currentTool === 'text') {
      const text = prompt('Enter text to insert:', 'Machine Learning Note');
      if (text) {
        this.ctx.fillStyle = this.currentColor;
        this.ctx.font = `${this.strokeWidth * 4 + 14}px Plus Jakarta Sans, sans-serif`;
        this.ctx.fillText(text, pos.x, pos.y);
        this.saveState();
      }
      this.isDrawing = false;
    }
  }

  onMouseMove(e) {
    const pos = this.getPointerPos(e);
    document.getElementById('cursor-pos').textContent = `X: ${Math.round(pos.x)}, Y: ${Math.round(pos.y)}`;

    if (!this.isDrawing) return;

    if (this.currentTool === 'brush') {
      this.currentPoints.push(pos);
      this.ctx.strokeStyle = this.currentColor;
      this.ctx.lineWidth = this.strokeWidth;
      this.ctx.globalCompositeOperation = 'source-over';

      if (this.currentPoints.length > 2) {
        const lastTwo = this.currentPoints.slice(-3);
        const xc = (lastTwo[1].x + lastTwo[2].x) / 2;
        const yc = (lastTwo[1].y + lastTwo[2].y) / 2;
        this.ctx.quadraticCurveTo(lastTwo[1].x, lastTwo[1].y, xc, yc);
        this.ctx.stroke();
      }
    } else if (this.currentTool === 'eraser') {
      this.ctx.globalCompositeOperation = 'destination-out';
      this.ctx.lineWidth = this.strokeWidth * 3;
      this.ctx.lineTo(pos.x, pos.y);
      this.ctx.stroke();
    } else {
      // Shape preview: restore snapshot, draw current shape
      this.ctx.putImageData(this.snapshot, 0, 0);
      this.ctx.globalCompositeOperation = 'source-over';
      this.ctx.strokeStyle = this.currentColor;
      this.ctx.fillStyle = this.currentColor;
      this.ctx.lineWidth = this.strokeWidth;

      if (this.currentTool === 'line') {
        this.ctx.beginPath();
        this.ctx.moveTo(this.startX, this.startY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
      } else if (this.currentTool === 'arrow') {
        this.drawArrow(this.startX, this.startY, pos.x, pos.y);
      } else if (this.currentTool === 'rectangle') {
        const w = pos.x - this.startX;
        const h = pos.y - this.startY;
        if (this.fillShape) {
          this.ctx.fillRect(this.startX, this.startY, w, h);
        } else {
          this.ctx.strokeRect(this.startX, this.startY, w, h);
        }
      } else if (this.currentTool === 'circle') {
        const radius = Math.hypot(pos.x - this.startX, pos.y - this.startY);
        this.ctx.beginPath();
        this.ctx.arc(this.startX, this.startY, radius, 0, 2 * Math.PI);
        if (this.fillShape) {
          this.ctx.fill();
        } else {
          this.ctx.stroke();
        }
      }
    }
  }

  onMouseUp() {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    this.ctx.closePath();
    this.saveState();
  }

  drawArrow(fromX, fromY, toX, toY) {
    const headLength = 16;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    this.ctx.beginPath();
    this.ctx.moveTo(fromX, fromY);
    this.ctx.lineTo(toX, toY);
    this.ctx.stroke();

    this.ctx.beginPath();
    this.ctx.moveTo(toX, toY);
    this.ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
    this.ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
    this.ctx.closePath();
    this.ctx.fillStyle = this.currentColor;
    this.ctx.fill();
  }

  createStickyNote(x, y, content = 'New Observation / Math Insight...') {
    const note = document.createElement('div');
    note.className = 'sticky-note';
    note.style.left = `${x}px`;
    note.style.top = `${y}px`;

    note.innerHTML = `
      <div class="sticky-header">
        <span>NOTE</span>
        <button class="sticky-close">&times;</button>
      </div>
      <textarea class="sticky-textarea">${content}</textarea>
    `;

    note.querySelector('.sticky-close').addEventListener('click', () => {
      note.remove();
    });

    // Make draggable
    let isDragging = false;
    let offsetX = 0;
    let offsetY = 0;

    note.querySelector('.sticky-header').addEventListener('mousedown', (e) => {
      isDragging = true;
      offsetX = e.clientX - note.getBoundingClientRect().left;
      offsetY = e.clientY - note.getBoundingClientRect().top;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      note.style.left = `${e.clientX - offsetX}px`;
      note.style.top = `${e.clientY - offsetY}px`;
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    this.stickyContainer.appendChild(note);
  }

  saveState() {
    this.historyIndex++;
    if (this.historyIndex < this.history.length) {
      this.history = this.history.slice(0, this.historyIndex);
    }
    this.history.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    if (this.history.length > this.maxHistory) {
      this.history.shift();
      this.historyIndex--;
    }
  }

  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      this.ctx.putImageData(this.history[this.historyIndex], 0, 0);
    }
  }

  redo() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++;
      this.ctx.putImageData(this.history[this.historyIndex], 0, 0);
    }
  }

  clearCanvas() {
    if (confirm('Clear the entire whiteboard canvas?')) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.stickyContainer.innerHTML = '';
      this.saveState();
    }
  }

  exportPNG() {
    // Create temporary export canvas combining canvas and dark background
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = this.canvas.width;
    exportCanvas.height = this.canvas.height;
    const expCtx = exportCanvas.getContext('2d');

    // Fill with canvas background
    expCtx.fillStyle = '#121622';
    expCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    expCtx.drawImage(this.canvas, 0, 0);

    const link = document.createElement('a');
    link.download = `auraboard_sketch_${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
    link.remove();
  }

  saveProjectJSON() {
    const data = {
      image: this.canvas.toDataURL(),
      timestamp: new Date().toISOString(),
      stickyNotes: Array.from(document.querySelectorAll('.sticky-note')).map(n => ({
        content: n.querySelector('.sticky-textarea').value,
        x: n.style.left,
        y: n.style.top
      }))
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `auraboard_project_${Date.now()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
    link.remove();
  }

  adjustZoom(delta) {
    this.scale = Math.max(0.4, Math.min(2.5, this.scale + delta));
    this.canvas.style.transform = `scale(${this.scale})`;
    this.canvas.style.transformOrigin = 'top left';
    document.getElementById('zoom-level').textContent = `${Math.round(this.scale * 100)}%`;
  }

  resetZoom() {
    this.scale = 1.0;
    this.canvas.style.transform = 'scale(1)';
    document.getElementById('zoom-level').textContent = '100%';
  }

  bindShortcuts() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        this.undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        this.redo();
      } else if (e.key.toLowerCase() === 'p') {
        this.setTool('brush');
      } else if (e.key.toLowerCase() === 'r') {
        this.setTool('rectangle');
      } else if (e.key.toLowerCase() === 'c') {
        this.setTool('circle');
      } else if (e.key.toLowerCase() === 'l') {
        this.setTool('line');
      } else if (e.key.toLowerCase() === 'a') {
        this.setTool('arrow');
      } else if (e.key.toLowerCase() === 'e') {
        this.setTool('eraser');
      } else if (e.key.toLowerCase() === 't') {
        this.setTool('text');
      } else if (e.key.toLowerCase() === 'n') {
        this.createStickyNote(window.innerWidth / 2 - 80, window.innerHeight / 2 - 60);
      }
    });
  }

  capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.whiteboard = new WhiteboardEngine();
});
