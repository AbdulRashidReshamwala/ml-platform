import sqlite3
from datetime import datetime
import pickle


def create_connection():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    return c, conn


def create_table(query):
    c, _ = create_connection()
    c.execute(query)


def insert_new_dataset(name, no_classes, no_images):
    c, conn = create_connection()
    c.execute(
        'INSERT INTO dataset (name, no_classes, no_images,date,status) VALUES( ?, ?, ?, ?, ?)', (name, no_classes, no_images, datetime.now(), 'created'))
    conn.commit()
    conn.close()


def get_dataset_info(name):
    c, conn = create_connection()
    c.execute('SELECT * FROM dataset WHERE name=? ',
              (name,))
    data = c.fetchone()
    conn.close()
    return data


def get_dataset_all():
    c, conn = create_connection()
    c.execute('SELECT name,no_classes,date FROM dataset')
    data = c.fetchall()
    conn.close()
    return data


def get_model_all():
    c, conn = create_connection()
    c.execute('SELECT name,architecture,dataset,date FROM model')
    data = c.fetchall()
    conn.close()
    return data


def update_status_dataset(name, status):
    c, conn = create_connection()
    c.execute(
        '''UPDATE dataset
        SET status=?
        WHERE name=?''', (status, name)
    )
    conn.commit()
    conn.close()


def update_status_scrapper(name, status):
    c, conn = create_connection()
    c.execute(
        '''UPDATE scrapper
        SET status=?
        WHERE name=?''', (status, name)
    )
    conn.commit()
    conn.close()


def update_status_model(name, status):
    c, conn = create_connection()
    print('uu')
    c.execute(
        '''UPDATE model
        SET status=?
        WHERE name=?''', (status, name)
    )
    conn.commit()
    conn.close()


def insert_new_model(arch, dataset, img_size, lr, epoch):
    d = datetime.now()
    name = (f'{arch}_{dataset}_{str(img_size)}_{epoch}_{str(d)}').replace(' ', '')
    c, conn = create_connection()
    c.execute(
        'INSERT INTO model (name,dataset,lr,img_size,epoch,architecture,date,status) VALUES( ?, ?, ?, ?, ?,?,?,?)', (name, dataset, lr, img_size, epoch, arch, d, 'queued'))
    conn.commit()
    conn.close()


def get_model_info(name):
    c, conn = create_connection()
    c.execute('SELECT * FROM model WHERE name=? ',
              (name,))
    data = c.fetchone()
    conn.close()
    return data


def insert_new_scrap(name, classes, num_images):
    c, conn = create_connection()
    c.execute(
        'INSERT INTO scrapper (name,num_images,classes,date,status) VALUES( ?, ?, ?, ?, ?)', (name, num_images, pickle.dumps(classes), len(classes), 'created'))
    conn.commit()
    conn.close()


dataset_query = '''CREATE TABLE dataset(
        id integer PRIMARY KEY AUTOINCREMENT,
        name text UNIQUE NOT NULL,
        no_classes integer NOT NULL,
        no_images integer NOT NULL,
        date timestamp NOT NULL,
        status text NOT NULL
)'''


model_query = '''CREATE TABLE model(
        id integer PRIMARY KEY AUTOINCREMENT,
        name text UNIQUE NOT NULL,
        dataset text NOT NULL,
        lr float NOT NULL,
        img_size integer NOT NULL,
        epoch integer NOT NULL,
        architecture text NOT NULL,
        date timestamp NOT NULL,
        status text NOT NULL,
        history blob
)'''


api_query = '''CREATE TABLE api(
        id integer PRIMARY KEY AUTOINCREMENT,
        name text UNIQUE NOT NULL,
        model text NOT NULL,
        url text,
        date timestamp NOT NULL,
        status text NOT NULL
)'''

scrapper_query = '''CREATE TABLE scrapper(
        id integer PRIMARY KEY AUTOINCREMENT,
        name text UNIQUE NOT NULL,
        num_images integer NOT NULL,
        classes BLOB NOT NULL,
        date timestamp NOT NULL,
        status text NOT NULL
)'''


if __name__ == "__main__":
    create_table(dataset_query)
    create_table(api_query)
    create_table(model_query)
    create_table(scrapper_query)
