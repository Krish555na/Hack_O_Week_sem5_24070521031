"""
Week 3: Automated Unit Tests for Python Essentials (OOP, Comprehensions & Generators)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
from datetime import date, timedelta
from inventory_oop import (
    LibraryCatalog, Book, DigitalCourse,
    ItemAlreadyBorrowedError, ItemNotBorrowedError,
    InvalidRatingError, ItemNotFoundError
)
from comprehensions_generators import (
    demonstrate_comprehensions, batch_generator,
    stream_filter_and_transform, fibonacci
)


class TestLibraryOOP(unittest.TestCase):
    def setUp(self):
        self.catalog = LibraryCatalog("Test Library")
        self.book1 = Book("b-101", "Introduction to Probability", "Bertsekas", 2008, "978-1886529236", 544, "Math")
        self.course1 = DigitalCourse("c-201", "Machine Learning Specialization", "Andrew Ng", 2022, 60.0, "Coursera")
        self.catalog.add(self.book1)
        self.catalog.add(self.course1)

    def test_catalog_length_and_membership(self):
        self.assertEqual(len(self.catalog), 2)
        self.assertIn("b-101", self.catalog)
        self.assertIn(self.book1, self.catalog)
        self.assertNotIn("non-existent-id", self.catalog)

    def test_indexing_and_lookup(self):
        self.assertEqual(self.catalog[0], self.book1)
        self.assertEqual(self.catalog["c-201"], self.course1)
        with self.assertRaises(ItemNotFoundError):
            _ = self.catalog["unknown-id"]

    def test_checkout_and_checkin_lifecycle(self):
        # Checkout
        due = self.book1.checkout("Alice Smith", days=10)
        self.assertTrue(self.book1.is_borrowed)
        self.assertEqual(self.book1.borrower, "Alice Smith")
        self.assertEqual(due, date.today() + timedelta(days=10))

        # Cannot borrow twice
        with self.assertRaises(ItemAlreadyBorrowedError):
            self.book1.checkout("Bob Jones")

        # Checkin without overdue
        fee = self.book1.checkin()
        self.assertEqual(fee, 0.0)
        self.assertFalse(self.book1.is_borrowed)

        # Cannot checkin if not borrowed
        with self.assertRaises(ItemNotBorrowedError):
            self.book1.checkin()

    def test_polymorphic_fee_calculation(self):
        # Book late fee (0.75 / day capped at 15.0)
        self.assertEqual(self.book1.calculate_late_fee(10), 7.50)
        self.assertEqual(self.book1.calculate_late_fee(30), 15.00)

        # Digital Course late fee (0.25 / day capped at 5.0)
        self.assertEqual(self.course1.calculate_late_fee(10), 2.50)
        self.assertEqual(self.course1.calculate_late_fee(40), 5.00)

    def test_rating_validation(self):
        self.book1.rating = 4.75
        self.assertEqual(self.book1.rating, 4.75)
        with self.assertRaises(InvalidRatingError):
            self.book1.rating = 6.5
        with self.assertRaises(InvalidRatingError):
            self.book1.rating = -1.0


class TestComprehensionsAndGenerators(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {"id": "1", "title": "Calculus Early Transcendentals", "author": "Stewart", "year": 2015, "genre": "Math", "rating": 4.9, "status": "available", "tags": ["math", "calculus"]},
            {"id": "2", "title": "Linear Algebra Done Right", "author": "Axler", "year": 2024, "genre": "Math", "rating": 4.8, "status": "available", "tags": ["math", "vectors"]},
            {"id": "3", "title": "Designing Data-Intensive Applications", "author": "Kleppmann", "year": 2017, "genre": "CS", "rating": 4.9, "status": "borrowed", "tags": ["cs", "distributed"]},
        ]

    def test_comprehensions_processing(self):
        res = demonstrate_comprehensions(self.sample_data)
        self.assertEqual(len(res["high_rated_titles"]), 2)
        self.assertIn("Calculus Early Transcendentals", res["high_rated_titles"])
        self.assertEqual(res["unique_genres"], {"Math", "CS"})
        self.assertEqual(res["genre_frequencies"]["Math"], 2)
        self.assertEqual(res["genre_frequencies"]["CS"], 1)

    def test_batch_generator(self):
        batches = list(batch_generator([1, 2, 3, 4, 5], batch_size=2))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    def test_memoized_fibonacci(self):
        self.assertEqual(fibonacci(10), 55)
        self.assertEqual(fibonacci(20), 6765)


if __name__ == "__main__":
    unittest.main()
