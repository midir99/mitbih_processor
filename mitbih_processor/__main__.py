# -*- coding: utf-8 -*-
from webclient import create_app


def main():
    print('This Python library facilitates the process of extracting and \n'
          'processing ECG signals from MIT-BIH Arrhythmia Database.\n'
          'Launching the web client...')

    app = create_app()
    print(
        'The app is running on http://127.0.0.1:5000/arrcls/signal_extraction')
    app.run('localhost', 5000)


if __name__ == '__main__':
    main()
