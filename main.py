# main.py
# Entry point

import faulthandler, sys, traceback
faulthandler.enable(all_threads=True)

def logCrash(excType, exc, tb):
    with open("crash.log", "w", encoding="utf-8") as f:
        f.write("".join(traceback.format_exception(excType, exc, tb)))

sys.excepthook = logCrash


from PyQt6.QtWidgets import QApplication
from window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(640, 580)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()