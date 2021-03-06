#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import subprocess
import time
import smtplib
import datetime

from email.mime.text import MIMEText


from_ = 'youngjun@sherlock.stanford.edu'
to_ = 'youngjun@stanford.edu'

userid = 'neosado@gmail.com'
password = os.getenv('PROCMONPWD')

while True:
    print(datetime.datetime.now())

    ps = subprocess.Popen('ps -ef | grep salloc | grep MineField | grep -v ps', shell = True, stdout = subprocess.PIPE)

    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()

    if output != "":
        print('MineField is running')

    else:
        msg = MIMEText("Hello!")
        msg['Subject'] = 'MineField is done'
        msg['From'] = from_
        msg['To'] = to_

        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(userid, password)
        server.sendmail(from_, to_, msg.as_string())
        server.quit()

        print('MineField is done')
        exit(1)

    sys.stdout.flush()

    time.sleep(3600)

    print()


