"""
Week 3: Python Essentials — OOP, Inheritance, Encapsulation, Protocols & Dunder Methods

This module implements a robust, object-oriented domain model for an educational resource
management system, demonstrating modern Python paradigms:
- Abstract Base Classes (abc.ABC, @abstractmethod)
- Encapsulation & validation via @property
- Class inheritance & Polymorphism
- Magic / Dunder methods (__str__, __repr__, __len__, __getitem__, __contains__, __iter__)
- Custom domain exceptions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Iterator, Any


# =====================================================================
# Custom Domain Exceptions
# =====================================================================
class LibraryError(Exception):
    """Base exception for the library domain."""
    pass


class ItemNotFoundError(LibraryError):
    """Raised when an requested item ID is not in catalog."""
    pass


class ItemAlreadyBorrowedError(LibraryError):
    """Raised when trying to borrow an already checked-out item."""
    pass


class ItemNotBorrowedError(LibraryError):
    """Raised when trying to return an item that was not borrowed."""
    pass


class InvalidRatingError(LibraryError):
    """Raised when a rating is outside acceptable [0.0, 5.0] bounds."""
    pass


# =====================================================================
# Domain Model Hierarchy
# =====================================================================
class LibraryItem(ABC):
    """
    Abstract base class for all catalog items.
    Enforces title, id, category, and fee calculation interface.
    """

    def __init__(self, item_id: str, title: str, creator: str, year: int, base_fee_per_day: float = 1.0):
        self._item_id = item_id
        self._title = title
        self._creator = creator
        self._year = year
        self._base_fee_per_day = max(0.0, float(base_fee_per_day))
        self._is_borrowed: bool = False
        self._borrower: Optional[str] = None
        self._due_date: Optional[date] = None
        self._borrow_count: int = 0
        self._rating: float = 5.0

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def creator(self) -> str:
        return self._creator

    @property
    def year(self) -> int:
        return self._year

    @property
    def is_borrowed(self) -> bool:
        return self._is_borrowed

    @property
    def borrower(self) -> Optional[str]:
        return self._borrower

    @property
    def due_date(self) -> Optional[date]:
        return self._due_date

    @property
    def rating(self) -> float:
        return self._rating

    @rating.setter
    def rating(self, value: float):
        if not (0.0 <= value <= 5.0):
            raise InvalidRatingError(f"Rating {value} must be between 0.0 and 5.0")
        self._rating = round(value, 2)

    @property
    def borrow_count(self) -> int:
        return self._borrow_count

    def checkout(self, borrower: str, days: int = 14) -> date:
        """Borrow item for specified duration."""
        if self._is_borrowed:
            raise ItemAlreadyBorrowedError(f"'{self.title}' is currently borrowed by {self._borrower}.")
        self._is_borrowed = True
        self._borrower = borrower
        self._due_date = date.today() + timedelta(days=days)
        self._borrow_count += 1
        return self._due_date

    def checkin(self) -> float:
        """Return item and return any overdue late fee accrued."""
        if not self._is_borrowed:
            raise ItemNotBorrowedError(f"'{self.title}' is not currently marked as borrowed.")

        late_fee = 0.0
        if self._due_date and date.today() > self._due_date:
            days_overdue = (date.today() - self._due_date).days
            late_fee = self.calculate_late_fee(days_overdue)

        self._is_borrowed = False
        self._borrower = None
        self._due_date = None
        return late_fee

    @abstractmethod
    def calculate_late_fee(self, days_overdue: int) -> float:
        """Polymorphic contract: subclasses compute fees according to asset rules."""
        pass

    @abstractmethod
    def get_summary(self) -> str:
        """Polymorphic contract: concise description string."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        return {
            "id": self._item_id,
            "title": self._title,
            "creator": self._creator,
            "year": self._year,
            "is_borrowed": self._is_borrowed,
            "borrower": self._borrower,
            "due_date": str(self._due_date) if self._due_date else None,
            "rating": self._rating,
            "borrow_count": self._borrow_count
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, LibraryItem):
            return self._item_id == other._item_id
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self._item_id}', title='{self._title}')>"

    def __str__(self) -> str:
        status = f"Borrowed by {self._borrower}" if self._is_borrowed else "Available"
        return f"{self.title} by {self.creator} ({self.year}) — [{status}]"


class Book(LibraryItem):
    """Physical or printed book item."""

    def __init__(self, item_id: str, title: str, author: str, year: int,
                 isbn: str, num_pages: int, genre: str = "Computer Science"):
        super().__init__(item_id, title, author, year, base_fee_per_day=0.75)
        self.isbn = isbn
        self.num_pages = num_pages
        self.genre = genre

    def calculate_late_fee(self, days_overdue: int) -> float:
        # Standard book: linear daily rate with $15 cap
        fee = days_overdue * self._base_fee_per_day
        return min(15.0, round(fee, 2))

    def get_summary(self) -> str:
        return f"Book: '{self.title}' | Genre: {self.genre} | {self.num_pages} pages | ISBN: {self.isbn}"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "type": "Book",
            "isbn": self.isbn,
            "num_pages": self.num_pages,
            "genre": self.genre
        })
        return d


class DigitalCourse(LibraryItem):
    """Interactive video or Jupyter courseware."""

    def __init__(self, item_id: str, title: str, instructor: str, year: int,
                 duration_hours: float, platform: str = "Online"):
        super().__init__(item_id, title, instructor, year, base_fee_per_day=0.25)
        self.duration_hours = duration_hours
        self.platform = platform

    def calculate_late_fee(self, days_overdue: int) -> float:
        # Digital licenses expire without physical loss; low flat penalty
        return min(5.0, round(days_overdue * 0.25, 2))

    def get_summary(self) -> str:
        return f"Course: '{self.title}' by {self.creator} ({self.duration_hours}h on {self.platform})"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "type": "DigitalCourse",
            "duration_hours": self.duration_hours,
            "platform": self.platform
        })
        return d


# =====================================================================
# Collection Container with Dunder / Magic Protocols
# =====================================================================
class LibraryCatalog:
    """
    Collection manager implementing standard container protocols:
    - len(catalog)
    - catalog[item_id] or catalog[index]
    - item in catalog
    - for item in catalog: ...
    """

    def __init__(self, name: str = "Main Catalog"):
        self.name = name
        self._items: List[LibraryItem] = []
        self._id_map: Dict[str, LibraryItem] = {}

    def add(self, item: LibraryItem) -> None:
        if item.item_id in self._id_map:
            raise ValueError(f"Item with ID '{item.item_id}' already exists in catalog.")
        self._items.append(item)
        self._id_map[item.item_id] = item

    def remove(self, item_id: str) -> LibraryItem:
        if item_id not in self._id_map:
            raise ItemNotFoundError(f"Item '{item_id}' not found.")
        item = self._id_map.pop(item_id)
        self._items.remove(item)
        return item

    def get(self, item_id: str) -> Optional[LibraryItem]:
        return self._id_map.get(item_id)

    # Magic Methods
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[LibraryItem]:
        return iter(self._items)

    def __getitem__(self, key: Any) -> LibraryItem:
        if isinstance(key, int):
            return self._items[key]
        elif isinstance(key, str):
            if key in self._id_map:
                return self._id_map[key]
            raise ItemNotFoundError(f"Key '{key}' not in catalog.")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return key in self._id_map
        elif isinstance(key, LibraryItem):
            return key in self._items
        return False

    def __repr__(self) -> str:
        return f"<LibraryCatalog(name='{self.name}', items={len(self._items)})>"


if __name__ == "__main__":
    print("=== Demo: Library OOP Domain Architecture ===")
    catalog = LibraryCatalog("AI & ML Research Library")

    b1 = Book("b-1", "Pattern Recognition & ML", "Christopher Bishop", 2006, "978-0387310732", 738, "Machine Learning")
    b2 = Book("b-2", "Deep Learning", "Goodfellow et al.", 2016, "978-0262035613", 800, "Machine Learning")
    c1 = DigitalCourse("c-1", "Fast.ai Practical Deep Learning", "Jeremy Howard", 2024, 24.5, "fast.ai")

    catalog.add(b1)
    catalog.add(b2)
    catalog.add(c1)

    print(f"Catalog size: {len(catalog)}")
    print(f"First item: {catalog[0]}")
    print(f"Direct ID lookup: {catalog['b-2']}")
    print(f"'b-1' in catalog: {'b-1' in catalog}")

    print("\nChecking out item...")
    due = catalog['b-1'].checkout("Dr. Alan Turing", days=7)
    print(f"Checked out until: {due}")
    print(f"Status: {catalog['b-1']}")
