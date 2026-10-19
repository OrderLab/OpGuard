import os
import tempfile
import unittest

from opguard.local_dump_manager import (
    LocalDumpManager,
    _periodic_sync_registry,
    _periodic_sync_registry_lock,
)


class TestPeriodicSyncServiceCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.managers = []
        with _periodic_sync_registry_lock:
            _periodic_sync_registry.clear()

    def tearDown(self):
        for manager in self.managers:
            try:
                manager.stop_periodic_sync()
            except Exception:
                pass
        with _periodic_sync_registry_lock:
            _periodic_sync_registry.clear()
        self.temp_dir.cleanup()

    def _new_manager(self, target_name: str) -> LocalDumpManager:
        target_dir = os.path.join(self.temp_dir.name, target_name)
        manager = LocalDumpManager(target_dir, sync_interval_sec=3600.0)
        self.managers.append(manager)
        return manager

    def test_skip_duplicate_periodic_sync_for_same_target(self):
        manager_a = self._new_manager("same_target")
        manager_b = self._new_manager("same_target")

        manager_a.start_periodic_sync()
        self.assertIsNotNone(manager_a._sync_thread)
        self.assertTrue(manager_a._sync_thread.is_alive())

        manager_b.start_periodic_sync()
        self.assertIsNone(manager_b._sync_thread)

        self.assertTrue(manager_a.stop_periodic_sync())

        manager_b.start_periodic_sync()
        self.assertIsNotNone(manager_b._sync_thread)
        self.assertTrue(manager_b._sync_thread.is_alive())

    def test_allow_periodic_sync_for_different_targets(self):
        manager_a = self._new_manager("target_a")
        manager_b = self._new_manager("target_b")

        manager_a.start_periodic_sync()
        manager_b.start_periodic_sync()

        self.assertIsNotNone(manager_a._sync_thread)
        self.assertTrue(manager_a._sync_thread.is_alive())
        self.assertIsNotNone(manager_b._sync_thread)
        self.assertTrue(manager_b._sync_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
