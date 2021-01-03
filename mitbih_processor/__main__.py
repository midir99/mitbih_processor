# -*- coding: utf-8 -*-
from webclient import create_app
import webbrowser
import threading


def main():
    print('This Python library facilitates the process of extracting and \n'
          'processing ECG signals from MIT-BIH Arrhythmia Database.\n'
          'Launching the web client...')
    app = create_app()
    url = 'http://127.0.0.1:5000/arrcls/signal_extraction'
    print(f'The app is running on {url}')
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    app.run('localhost', 5000, debug=False)


if __name__ == '__main__':
    main()
