
const SEED_BOOKS = [
  {
    id: "book-1",
    title: "Deep Learning",
    author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
    genre: "Machine Learning",
    year: 2016,
    isbn: "978-0262035613",
    rating: 4.9,
    copies: 4,
    status: "available",
    summary: "The definitive textbook on deep learning covering linear algebra, probability, deep networks, regularization, and generative models.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-2",
    title: "Pattern Recognition and Machine Learning",
    author: "Christopher M. Bishop",
    genre: "Machine Learning",
    year: 2006,
    isbn: "978-0387310732",
    rating: 4.8,
    copies: 2,
    status: "available",
    summary: "Comprehensive introduction to the fields of pattern recognition and machine learning from a Bayesian perspective.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-3",
    title: "Artificial Intelligence: A Modern Approach",
    author: "Stuart Russell, Peter Norvig",
    genre: "Artificial Intelligence",
    year: 2020,
    isbn: "978-0134610993",
    rating: 4.9,
    copies: 3,
    status: "borrowed",
    summary: "The most widely used AI textbook, exploring intelligent agents, search algorithms, logic, probabilistic reasoning, and robotics.",
    borrowedBy: "Sarah Chen (ID: AI-101)",
    dueDate: "2026-09-22"
  },
  {
    id: "book-4",
    title: "Introduction to Linear Algebra",
    author: "Gilbert Strang",
    genre: "Mathematics",
    year: 2016,
    isbn: "978-0980232776",
    rating: 4.9,
    copies: 5,
    status: "available",
    summary: "Standard MIT reference for vectors, matrices, subspaces, eigenvalues, singular value decomposition, and transformations.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-5",
    title: "Introduction to Algorithms (CLRS)",
    author: "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein",
    genre: "Computer Science",
    year: 2022,
    isbn: "978-0262046305",
    rating: 4.8,
    copies: 3,
    status: "available",
    summary: "Rigorous and comprehensive coverage of data structures, dynamic programming, greedy algorithms, graph algorithms, and NP-completeness.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-6",
    title: "Python for Data Analysis",
    author: "Wes McKinney",
    genre: "Data Science",
    year: 2022,
    isbn: "978-1098104030",
    rating: 4.7,
    copies: 4,
    status: "available",
    summary: "Practical guide by the creator of pandas for manipulating, processing, cleaning, and crunching datasets in Python.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-7",
    title: "Mathematics for Machine Learning",
    author: "Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong",
    genre: "Mathematics",
    year: 2020,
    isbn: "978-1108455145",
    rating: 4.8,
    copies: 2,
    status: "available",
    summary: "Bridges mathematical concepts (linear algebra, analytic geometry, matrix decompositions, vector calculus) with modern ML models.",
    borrowedBy: null,
    dueDate: null
  },
  {
    id: "book-8",
    title: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
    author: "Aurélien Géron",
    genre: "Machine Learning",
    year: 2022,
    isbn: "978-1098125974",
    rating: 4.9,
    copies: 5,
    status: "borrowed",
    summary: "End-to-end practical walkthrough of building intelligent systems, training classifiers, and deploying deep neural networks.",
    borrowedBy: "Marcus Thorne (ID: CS-509)",
    dueDate: "2026-09-18"
  }
];

class LibraryStore {
  constructor() {
    this.STORAGE_KEY = "auralib_books_v1";
    this.HISTORY_KEY = "auralib_history_v1";
    this.books = this.loadBooks();
    this.history = this.loadHistory();
  }

  loadBooks() {
    const raw = localStorage.getItem(this.STORAGE_KEY);
    if (!raw) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(SEED_BOOKS));
      return SEED_BOOKS;
    }
    try {
      return JSON.parse(raw);
    } catch {
      return SEED_BOOKS;
    }
  }

  saveBooks() {
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.books));
  }

  loadHistory() {
    const raw = localStorage.getItem(this.HISTORY_KEY);
    return raw ? JSON.parse(raw) : [
      { id: 'h-1', action: 'System Initialized', detail: 'Loaded 8 foundational library texts', timestamp: new Date().toISOString() }
    ];
  }

  logHistory(action, detail) {
    const entry = {
      id: 'h-' + Date.now(),
      action,
      detail,
      timestamp: new Date().toISOString()
    };
    this.history.unshift(entry);
    if (this.history.length > 50) this.history.pop();
    localStorage.setItem(this.HISTORY_KEY, JSON.stringify(this.history));
  }

  addBook(book) {
    const newBook = {
      ...book,
      id: 'book-' + Date.now(),
      status: 'available',
      borrowedBy: null,
      dueDate: null
    };
    this.books.unshift(newBook);
    this.saveBooks();
    this.logHistory('Added Book', `"${newBook.title}" added to catalog`);
    return newBook;
  }

  updateBook(id, updatedFields) {
    const idx = this.books.findIndex(b => b.id === id);
    if (idx !== -1) {
      this.books[idx] = { ...this.books[idx], ...updatedFields };
      this.saveBooks();
      this.logHistory('Updated Book', `"${this.books[idx].title}" details updated`);
      return this.books[idx];
    }
    return null;
  }

  deleteBook(id) {
    const book = this.books.find(b => b.id === id);
    if (book) {
      this.books = this.books.filter(b => b.id !== id);
      this.saveBooks();
      this.logHistory('Deleted Book', `"${book.title}" removed from catalog`);
    }
  }

  borrowBook(id, borrowerName, days = 14) {
    const book = this.books.find(b => b.id === id);
    if (book && book.status === 'available') {
      const due = new Date();
      due.setDate(due.getDate() + parseInt(days, 10));
      book.status = 'borrowed';
      book.borrowedBy = borrowerName;
      book.dueDate = due.toISOString().split('T')[0];
      this.saveBooks();
      this.logHistory('Borrowed Book', `"${book.title}" loaned to ${borrowerName} (Due: ${book.dueDate})`);
      return true;
    }
    return false;
  }

  returnBook(id) {
    const book = this.books.find(b => b.id === id);
    if (book && book.status === 'borrowed') {
      const prevBorrower = book.borrowedBy;
      book.status = 'available';
      book.borrowedBy = null;
      book.dueDate = null;
      this.saveBooks();
      this.logHistory('Returned Book', `"${book.title}" checked back in from ${prevBorrower}`);
      return true;
    }
    return false;
  }
}

// UI Controller
class LibraryApp {
  constructor() {
    this.store = new LibraryStore();
    this.currentFilterGenre = 'all';
    this.currentFilterStatus = 'all';
    this.currentSort = 'title-asc';
    this.searchQuery = '';

    this.bindDOMElements();
    this.bindEvents();
    this.render();
  }

  bindDOMElements() {
    // Stats
    this.countTotal = document.getElementById('count-total');
    this.countAvailable = document.getElementById('count-available');
    this.countBorrowed = document.getElementById('count-borrowed');
    this.countCategories = document.getElementById('count-categories');
    this.filteredCount = document.getElementById('filtered-count');

    // Controls
    this.searchInput = document.getElementById('search-input');
    this.genrePillsContainer = document.getElementById('genre-pills-container');
    this.statusFilter = document.getElementById('status-filter');
    this.sortSelect = document.getElementById('sort-select');
    this.booksGrid = document.getElementById('books-grid');
    this.emptyState = document.getElementById('empty-state');
    this.btnResetFilters = document.getElementById('btn-reset-filters');

    // Modals
    this.bookModal = document.getElementById('book-modal');
    this.modalTitle = document.getElementById('modal-title');
    this.bookForm = document.getElementById('book-form');
    this.formBookId = document.getElementById('form-book-id');
    this.btnAddBook = document.getElementById('btn-add-book');
    this.btnCloseModal = document.getElementById('btn-close-modal');
    this.btnCancelForm = document.getElementById('btn-cancel-form');

    // Borrow Modal
    this.borrowModal = document.getElementById('borrow-modal');
    this.borrowForm = document.getElementById('borrow-form');
    this.borrowBookId = document.getElementById('borrow-book-id');
    this.borrowBookPreview = document.getElementById('borrow-book-preview');
    this.borrowerNameInput = document.getElementById('borrower-name');
    this.borrowDaysSelect = document.getElementById('borrow-days');
    this.btnCloseBorrow = document.getElementById('btn-close-borrow');
    this.btnCancelBorrow = document.getElementById('btn-cancel-borrow');

    // Drawer & Actions
    this.historyDrawer = document.getElementById('history-drawer');
    this.btnActivity = document.getElementById('btn-activity');
    this.btnCloseDrawer = document.getElementById('btn-close-drawer');
    this.activityLogList = document.getElementById('activity-log-list');
    this.btnClearHistory = document.getElementById('btn-clear-history');
    this.btnExport = document.getElementById('btn-export');
  }

  bindEvents() {
    // Search
    this.searchInput.addEventListener('input', (e) => {
      this.searchQuery = e.target.value.toLowerCase().trim();
      this.render();
    });

    // Keyboard shortcut '/'
    window.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== this.searchInput) {
        e.preventDefault();
        this.searchInput.focus();
      }
    });

    // Genre filter pills
    this.genrePillsContainer.addEventListener('click', (e) => {
      if (e.target.classList.contains('pill')) {
        this.genrePillsContainer.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        e.target.classList.add('active');
        this.currentFilterGenre = e.target.dataset.genre;
        this.render();
      }
    });

    // Dropdowns
    this.statusFilter.addEventListener('change', (e) => {
      this.currentFilterStatus = e.target.value;
      this.render();
    });

    this.sortSelect.addEventListener('change', (e) => {
      this.currentSort = e.target.value;
      this.render();
    });

    this.btnResetFilters.addEventListener('click', () => {
      this.searchQuery = '';
      this.searchInput.value = '';
      this.currentFilterGenre = 'all';
      this.genrePillsContainer.querySelectorAll('.pill').forEach(p => {
        p.classList.toggle('active', p.dataset.genre === 'all');
      });
      this.currentFilterStatus = 'all';
      this.statusFilter.value = 'all';
      this.render();
    });

    // Add Book Modal
    this.btnAddBook.addEventListener('click', () => this.openBookModal());
    this.btnCloseModal.addEventListener('click', () => this.closeBookModal());
    this.btnCancelForm.addEventListener('click', () => this.closeBookModal());
    this.bookForm.addEventListener('submit', (e) => this.handleSaveBook(e));

    // Borrow Modal
    this.btnCloseBorrow.addEventListener('click', () => this.closeBorrowModal());
    this.btnCancelBorrow.addEventListener('click', () => this.closeBorrowModal());
    this.borrowForm.addEventListener('submit', (e) => this.handleConfirmBorrow(e));

    // Drawer
    this.btnActivity.addEventListener('click', () => this.openHistoryDrawer());
    this.btnCloseDrawer.addEventListener('click', () => this.closeHistoryDrawer());
    this.btnClearHistory.addEventListener('click', () => {
      this.store.history = [];
      localStorage.setItem(this.store.HISTORY_KEY, JSON.stringify([]));
      this.renderHistory();
    });

    // Export
    this.btnExport.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.store.books, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `auralib_catalog_${new Date().toISOString().split('T')[0]}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  getFilteredAndSortedBooks() {
    return this.store.books.filter(book => {
      const matchesSearch =
        book.title.toLowerCase().includes(this.searchQuery) ||
        book.author.toLowerCase().includes(this.searchQuery) ||
        book.isbn.toLowerCase().includes(this.searchQuery);

      const matchesGenre = this.currentFilterGenre === 'all' || book.genre.toLowerCase() === this.currentFilterGenre.toLowerCase();
      const matchesStatus = this.currentFilterStatus === 'all' || book.status === this.currentFilterStatus;

      return matchesSearch && matchesGenre && matchesStatus;
    }).sort((a, b) => {
      switch (this.currentSort) {
        case 'title-asc': return a.title.localeCompare(b.title);
        case 'title-desc': return b.title.localeCompare(a.title);
        case 'rating-desc': return b.rating - a.rating;
        case 'year-desc': return b.year - a.year;
        default: return 0;
      }
    });
  }

  render() {
    this.renderStats();
    const filtered = this.getFilteredAndSortedBooks();
    this.filteredCount.textContent = `Showing ${filtered.length} of ${this.store.books.length} books`;

    if (filtered.length === 0) {
      this.booksGrid.innerHTML = '';
      this.emptyState.classList.remove('hidden');
    } else {
      this.emptyState.classList.add('hidden');
      this.booksGrid.innerHTML = filtered.map(book => this.createBookCardHTML(book)).join('');
      this.attachBookCardActions();
    }
  }

  createBookCardHTML(book) {
    const isAvail = book.status === 'available';
    return `
      <article class="book-card" id="card-${book.id}">
        <div class="book-top">
          <span class="genre-badge">${this.escapeHTML(book.genre)}</span>
          <span class="status-badge ${isAvail ? 'available' : 'borrowed'}">
            ${isAvail ? '● Available' : '● Borrowed'}
          </span>
        </div>

        <div class="book-info">
          <h3 class="book-title">${this.escapeHTML(book.title)}</h3>
          <p class="book-author">by ${this.escapeHTML(book.author)}</p>
        </div>

        <div class="book-details">
          <div class="book-meta">
            <span class="meta-isbn">ISBN: ${this.escapeHTML(book.isbn)}</span>
            <span class="meta-rating">★ ${book.rating}</span>
          </div>
          <p class="book-desc">${this.escapeHTML(book.summary || 'No summary available.')}</p>

          ${!isAvail && book.borrowedBy ? `
            <div class="borrow-info-box">
              <span>👤 ${this.escapeHTML(book.borrowedBy)}</span>
              <span>Due: ${book.dueDate || 'N/A'}</span>
            </div>
          ` : ''}
        </div>

        <div class="book-card-actions">
          <div style="display:flex; gap:0.4rem;">
            ${isAvail ? `
              <button class="btn btn-primary btn-sm btn-borrow" data-id="${book.id}">
                Borrow
              </button>
            ` : `
              <button class="btn btn-secondary btn-sm btn-return" data-id="${book.id}">
                Return
              </button>
            `}
            <button class="btn btn-secondary btn-sm btn-edit" data-id="${book.id}">Edit</button>
          </div>
          <button class="btn btn-danger btn-sm btn-delete" data-id="${book.id}" title="Remove Book">🗑</button>
        </div>
      </article>
    `;
  }

  attachBookCardActions() {
    this.booksGrid.querySelectorAll('.btn-borrow').forEach(btn => {
      btn.addEventListener('click', () => this.openBorrowModal(btn.dataset.id));
    });

    this.booksGrid.querySelectorAll('.btn-return').forEach(btn => {
      btn.addEventListener('click', () => {
        this.store.returnBook(btn.dataset.id);
        this.render();
      });
    });

    this.booksGrid.querySelectorAll('.btn-edit').forEach(btn => {
      btn.addEventListener('click', () => this.openBookModal(btn.dataset.id));
    });

    this.booksGrid.querySelectorAll('.btn-delete').forEach(btn => {
      btn.addEventListener('click', () => {
        const book = this.store.books.find(b => b.id === btn.dataset.id);
        if (confirm(`Are you sure you want to delete "${book ? book.title : 'this book'}"?`)) {
          this.store.deleteBook(btn.dataset.id);
          this.render();
        }
      });
    });
  }

  renderStats() {
    const total = this.store.books.length;
    const available = this.store.books.filter(b => b.status === 'available').length;
    const borrowed = total - available;
    const genres = new Set(this.store.books.map(b => b.genre)).size;

    this.countTotal.textContent = total;
    this.countAvailable.textContent = available;
    this.countBorrowed.textContent = borrowed;
    this.countCategories.textContent = genres;
  }

  openBookModal(bookId = null) {
    this.bookForm.reset();
    if (bookId) {
      const book = this.store.books.find(b => b.id === bookId);
      if (book) {
        this.modalTitle.textContent = "Edit Book";
        this.formBookId.value = book.id;
        document.getElementById('form-title').value = book.title;
        document.getElementById('form-author').value = book.author;
        document.getElementById('form-genre').value = book.genre;
        document.getElementById('form-year').value = book.year;
        document.getElementById('form-isbn').value = book.isbn;
        document.getElementById('form-rating').value = book.rating;
        document.getElementById('form-copies').value = book.copies || 1;
        document.getElementById('form-summary').value = book.summary || '';
      }
    } else {
      this.modalTitle.textContent = "Add New Book";
      this.formBookId.value = '';
    }
    this.bookModal.classList.remove('hidden');
    document.getElementById('form-title').focus();
  }

  closeBookModal() {
    this.bookModal.classList.add('hidden');
  }

  handleSaveBook(e) {
    e.preventDefault();
    const id = this.formBookId.value;
    const bookData = {
      title: document.getElementById('form-title').value.trim(),
      author: document.getElementById('form-author').value.trim(),
      genre: document.getElementById('form-genre').value,
      year: parseInt(document.getElementById('form-year').value, 10),
      isbn: document.getElementById('form-isbn').value.trim(),
      rating: parseFloat(document.getElementById('form-rating').value) || 5.0,
      copies: parseInt(document.getElementById('form-copies').value, 10) || 1,
      summary: document.getElementById('form-summary').value.trim()
    };

    if (id) {
      this.store.updateBook(id, bookData);
    } else {
      this.store.addBook(bookData);
    }

    this.closeBookModal();
    this.render();
  }

  openBorrowModal(bookId) {
    const book = this.store.books.find(b => b.id === bookId);
    if (!book) return;
    this.borrowBookId.value = book.id;
    this.borrowBookPreview.innerHTML = `
      <div style="font-weight:700; margin-bottom:0.25rem;">${this.escapeHTML(book.title)}</div>
      <div style="font-size:0.8rem; color:var(--text-secondary);">Author: ${this.escapeHTML(book.author)}</div>
      <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.75rem;">ISBN: ${this.escapeHTML(book.isbn)}</div>
    `;
    this.borrowerNameInput.value = '';
    this.borrowModal.classList.remove('hidden');
    this.borrowerNameInput.focus();
  }

  closeBorrowModal() {
    this.borrowModal.classList.add('hidden');
  }

  handleConfirmBorrow(e) {
    e.preventDefault();
    const id = this.borrowBookId.value;
    const borrower = this.borrowerNameInput.value.trim();
    const days = this.borrowDaysSelect.value;
    if (id && borrower) {
      this.store.borrowBook(id, borrower, days);
      this.closeBorrowModal();
      this.render();
    }
  }

  openHistoryDrawer() {
    this.renderHistory();
    this.historyDrawer.classList.remove('hidden');
  }

  closeHistoryDrawer() {
    this.historyDrawer.classList.add('hidden');
  }

  renderHistory() {
    if (this.store.history.length === 0) {
      this.activityLogList.innerHTML = `<div style="text-align:center; color:var(--text-muted); padding:2rem 0;">No activities logged yet.</div>`;
      return;
    }
    this.activityLogList.innerHTML = this.store.history.map(item => `
      <div class="history-item">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong>${this.escapeHTML(item.action)}</strong>
          <span class="history-time">${new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div style="color:var(--text-secondary);">${this.escapeHTML(item.detail)}</div>
      </div>
    `).join('');
  }

  escapeHTML(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Instantiate on load
document.addEventListener('DOMContentLoaded', () => {
  window.app = new LibraryApp();
});
