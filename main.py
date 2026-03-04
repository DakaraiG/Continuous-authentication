# main.py
# Entry point

import faulthandler, sys, traceback, logging

# In windowed (no-console) builds sys.stderr/stdout are None; redirect faulthandler to a file
_faultlog = open("fault.log", "w")
faulthandler.enable(file=_faultlog, all_threads=True)

def logCrash(excType, exc, tb):
    logging.critical("Unhandled exception", exc_info=(excType, exc, tb))
    with open("crash.log", "w", encoding="utf-8") as f:
        f.write("".join(traceback.format_exception(excType, exc, tb)))

sys.excepthook = logCrash

from app_log import setupLogging
setupLogging()

from PyQt6.QtWidgets import QApplication
from window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(640, 580)
    window.show()
    logging.info("Main window displayed")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()