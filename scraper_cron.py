from icrawler.builtin import BingImageCrawler
import dbutility
import pickle
import os


while 1:
    c, conn = dbutility.create_connection()
    c.execute('SELECT * FROM scrapper WHERE status=? ',
              ('created',))
    data = c.fetchall()
    # print(data)
    for task in data:
        dbutility.update_status_scrapper(task[0], 'started')
        classes = pickle.loads(task[3])

        for c in classes:
            bing_crawler = BingImageCrawler(downloader_threads=6, storage={
                'root_dir': f'static/datasets/{task[1]}/{c}'})
            bing_crawler.crawl(keyword=c, filters=None,
                               offset=0, max_num=int(task[2]))
        num_images = 0
        num_classes = 0
        dataset_path = f'static/datasets/{task[1]}'
        for clx in os.listdir(dataset_path):
            num_classes += 1
            num_images += len(os.listdir(os.path.join(dataset_path, clx)))
        dbutility.update_status_scrapper(task[1], 'completed')
        dbutility.insert_new_dataset(task[1], num_classes, num_images)
    conn.close()
