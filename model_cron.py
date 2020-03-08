import dbutility
import pickle
import os
from akhri import run

while 1:
    c, conn = dbutility.create_connection()
    c.execute('SELECT * FROM model WHERE status=? ',
              ('queued',))
    data = c.fetchall()
    # print(data)
    for task in data:
        print(task[1])
        run(f'static/datasets/{task[2]}', task[4], task[6], task[5], task[1])
        dbutility.update_status_model(task[1], 'completed')

    conn.close()
# f'static/datasets/{task[2]}'
