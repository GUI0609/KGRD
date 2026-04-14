"""Neo4j 官方驱动轻量封装。"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Neo4jClient:
    """管理 Driver 与 session（含 database 名称）。"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self.database = database
        self._driver: Driver | None = None

    def connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            logger.info("Neo4j connectivity OK: %s db=%s", self._uri, self.database)
        except Exception as e:
            logger.error("Neo4j connection failed: %s", e)
            raise

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def read(self, fn: Callable[[Any], T]) -> T:
        if not self._driver:
            raise RuntimeError("Driver not connected")
        with self._driver.session(database=self.database) as session:
            return session.execute_read(fn)

    def write(self, fn: Callable[[Any], T]) -> T:
        if not self._driver:
            raise RuntimeError("Driver not connected")
        with self._driver.session(database=self.database) as session:
            return session.execute_write(fn)

    def run_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        def work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(cypher, params or {})
            return [r.data() for r in result]

        return self.read(work)

    def iter_run(self, cypher: str, params: dict[str, Any] | None = None):
        """
        Stream records from a read query. Session stays open until iteration completes.

        Use for large exports to avoid loading the full result into RAM at once.
        """
        if not self._driver:
            raise RuntimeError("Driver not connected")
        session = self._driver.session(database=self.database)
        try:
            result = session.run(cypher, params or {})
            for record in result:
                yield record
        finally:
            session.close()
