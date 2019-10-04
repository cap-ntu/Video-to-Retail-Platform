import os
import os.path as osp
import pafy
import json


class MSR_VTT(object):
    """ http://ms-multimedia-challenge.com/2017/dataset

    Download and Index MSR_VTT dataset.  (About 500GB)

    Please organize the basic folder like this:
        |- root_dir/
            |- scripts/
                |- MSR _VTT.py
                |- videodatainfo_2017.json
                |- test_videodatainfo_nosen_2017.json
                |- category.txt
    The video data and index file will be stored under root_dir.
    Or you can modify the paths in __init__ as you want.
    """

    def __init__(self):

        self.root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))

        # data_dir: contains the video data (.mp4).
        self.data_dir = osp.join(self.root_dir, 'data')
        # json_dir: contains the json output file.
        self.json_dir = osp.join(self.root_dir, 'json')
        # scripts_dir: contains the origin data file.
        self.scripts_dir = osp.join(self.root_dir, 'scripts')

        self.train_dir = osp.join(self.data_dir, 'train')
        self.test_dir = osp.join(self.data_dir, 'test')

        self.make_dir(self.data_dir)
        self.make_dir(self.json_dir)
        self.make_dir(self.train_dir)
        self.make_dir(self.test_dir)

        self.train_file = osp.join(self.scripts_dir, 'videodatainfo_2017.json')
        self.test_file = osp.join(self.scripts_dir, 'test_videodatainfo_nosen_2017.json')
        self.category_file = osp.join(self.scripts_dir, 'category.txt')

        self.check_file(self.train_file)
        self.check_file(self.test_file)
        self.check_file(self.category_file)

        self.train_info = None
        self.train_video = None
        self.train_sentence = None

        self.test_info = None
        self.test_video = None
        self.test_sentence = None

        self.category = None

        self.download_list = {}

    @staticmethod
    def make_dir(dir_path):
        """ Check if dir exists."""
        if not osp.isdir(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def check_file(file_path):
        """ Check if file exists."""
        if not osp.isfile(file_path):
            print('{} not found!'.format(file_path.split('/')[-1]))

    def load_train_data(self):
        """ Load data from train data file."""
        with open(self.train_file, 'r') as f:
            data = json.load(f)
            if not self.train_info:
                self.train_info = data["info"]
            if not self.train_video:
                self.train_video = data["videos"]
            if not self.train_sentence:
                self.train_sentence = data["sentences"]
        """ Show training set information"""
        print("-------------------MSR_VTT TRAIN----------------------")
        for key in self.train_info.keys():
            print("%-13s:\t%-40s" % (key, self.train_info[key]))
        print("------------------------------------------------------")
        print("%-7d%-10s" % (len(self.train_video), "Videos"))
        print("%-7d%-10s" % (len(self.train_sentence), "Sentences"))

    def split(self, size=1000):
        """ Split the video list into small json files."""
        count = size
        pkg_id = 0
        buffer = []
        for video in self.train_video:
            buffer.append(video)
            count -= 1
            if count == 0:
                file_name = 'data_' + str(pkg_id) + '.json'
                file_path = osp.join(self.json_dir, file_name)
                self.download_list[pkg_id] = file_path
                with open(file_path, 'w') as f:
                    json.dump(buffer, f)
                buffer = []
                pkg_id += 1
                count = size
        if buffer:
            file_name = 'data_' + str(pkg_id) + '.json'
            file_path = osp.join(self.json_dir, file_name)
            self.download_list[pkg_id] = file_path
            with open(file_path, 'w') as f:
                json.dump(buffer, f)
        print("Finish split video list into %d json files." % pkg_id)
        print("-------------------Download List----------------------")
        for item in self.download_list.keys():
            print("{}:{}".format(item, self.download_list[item]))

    def get_download_list(self):
        return self.download_list

    def do_download_train(self, json_path):
        """ Download videos using urls in given file."""
        if osp.isfile(json_path):
            name = json_path.split('/')[-1]
            store_invalid = []
            file_invalid = "invalid" + name
            store_downloaded = []
            file_downloaded = "downloaded" + name
            counter = 0
            with open(json_path, 'r') as f:
                data = json.load(f)
            num = len(data)
            print("{} videos to be downloaded in {}.".format(num, name))
            for video in data:
                url = video['url'].split('=')[1]
                counter += 1
                print("\tDownloading video {} ...".format(video['video_id']), end='')
                path = self.downloader(url, self.train_dir)
                if path:
                    video['path'] = path
                    store_downloaded.append(video)
                    print("\tSuccess. {}% completed.".format(round(counter / num * 100, 2)))
                else:
                    store_invalid.append(video)
                    print("\tFailed. {}% completed.".format(round(counter / num * 100, 2)))

            print("Finish downloading. \t{} success. \t{} failed.".format(len(store_downloaded), len(store_invalid)))
            print("Dumping files ... ")
            with open(osp.join(self.json_dir, file_invalid), 'w') as f:
                print("\tDumping invalid videos ... ", end='')
                json.dump(store_invalid, f)
                del store_invalid
                print("\tDone.")
            with open(osp.join(self.json_dir, file_downloaded), 'w') as f:
                print("\tDumping downloaded videos ...", end='')
                json.dump(store_downloaded, f)
                del store_downloaded
                print("\tDone.")
        else:
            print("Invalid path! {}".format(json_path))

    @staticmethod
    def downloader(url, path):
        """ Based on pafy."""
        try:
            video_instance = pafy.new(url)
        except Exception:
            return False
        video_path = osp.join(path, video_instance.title)
        if osp.isfile(video_path):
            return video_path
        else:
            try:
                # https://stackoverflow.com/questions/40713268/download-youtube-video-using-python-to-a-certain-directory
                video_path = video_instance.getbest(preftype="mp4").download(filepath=path, quiet=True)
                return video_path
            except Exception:
                return False

    def indexer(self):
        print("Combine all downloaded data file ... ", end='')
        downloaded_data = []
        downloaded_file = [f for f in os.listdir(self.json_dir) if f.split('_')[0] == 'downloadeddata']
        for file in downloaded_file:
            with open(osp.join(self.json_dir, file), 'r') as f:
                downloaded_data += json.load(f)
        print("\tDone.")
        print("Got {} videos".format(len(downloaded_data)))
        print("Start indexing MSR-VTT dataset ...")
        dataset = []
        for caption in self.train_sentence:
            for video in downloaded_data:
                if video["video_id"] == caption["video_id"]:
                    try:
                        video["count"] += 1
                    except KeyError:
                        video["count"] = 1
                    temp = {
                        "sen_id": caption["sen_id"],
                        "caption": caption["caption"],
                        "category": video["category"],
                        "start_time": video["start time"],
                        "end_time": video["end time"],
                        "path": video["path"]
                    }
                    dataset.append(temp)
        print("\tDone.")
        print("Checking video captions ... ")
        counter = 0
        for video in downloaded_data:
            if video["count"] is not 20:
                counter += 1
        print("\t{} videos have less than 20 sentences.".format(counter))
        print("\t{} captions.".format(len(dataset)))
        print("Dumping ... ", end='')
        with open(osp.join(self.root_dir, "captions.json"), 'w') as f:
            json.dump(dataset, f)
        with open(osp.join(self.json_dir, "combined_data.json"), 'w') as f:
            json.dump(downloaded_data, f)
        print("\tDone.")

    def load_test_data(self):
        """ Load data from test data file."""
        with open(self.test_file, 'r') as f:
            data = json.load(f)
            if not self.test_info:
                self.test_info = data["info"]
            if not self.test_video:
                self.test_video = data["videos"]
            if not self.test_sentence:
                self.test_sentence = data["sentences"]
        """ Show training set information"""
        print("------------------MSR_VTT TEST------------------------")
        for key in self.test_info.keys():
            print("%-13s:\t%-40s" % (key, self.test_info[key]))
        print("------------------------------------------------------")
        print("%-7d%-10s" % (len(self.test_video), "Videos"))
        print("%-7d%-10s" % (len(self.test_sentence), "Sentences"))

    def do_download_test(self):
        """ Download videos using urls in given file."""
        store_invalid = []
        file_invalid = "invalid_test.json"
        store_downloaded = []
        file_downloaded = "test_data.json"
        counter = 0
        num = len(self.test_video)
        print("{} videos will be download in test set.".format(num))
        for video in self.test_video:
            url = video['url'].split('=')[1]
            counter += 1
            print("\tDownloading video {} ...".format(video['video_id']), end='')
            # path = self.downloader(url, self.test_dir)
            path = "666"
            if path:
                video['path'] = path
                store_downloaded.append(video)
                print("\tSuccess. {}% completed.".format(round(counter / num * 100, 2)))
            else:
                store_invalid.append(video)
                print("\tFailed. {}% completed.".format(round(counter / num * 100, 2)))

        print("Finish downloading. \t{} success. \t{} failed.".format(len(store_downloaded), len(store_invalid)))
        print("Dumping files ... ")
        with open(osp.join(self.json_dir, file_invalid), 'w') as f:
            print("\tDumping invalid videos ... ", end='')
            json.dump(store_invalid, f)
            del store_invalid
            print("\tDone.")
        with open(osp.join(self.root_dir, file_downloaded), 'w') as f:
            print("\tDumping downloaded videos ...", end='')
            json.dump(store_downloaded, f)
            del store_downloaded
            print("\tDone.")

    def load_category_data(self):
        """"""
        category_dict = {}
        with open(self.category_file, 'r') as f:
            categories = f.readlines()
            for category in categories:
                name, code = category.strip().split('\t')
                category_dict[code] = int(name)
            self.category = category_dict

    def get_category_name(self, category_id):
        """"""
        if isinstance(category_id, str):
            category_id = int(category_id)
        elif isinstance(category_id, int):
            pass
        else:
            print("Code should be str or int.")
            return
        if self.category:
            return self.category[category_id]
        else:
            self.load_category_data()
            return self.category[category_id]


if __name__ == '__main__':
    """ How to use."""
    # 1. Init
    vtt = MSR_VTT()

    # 2. Download and Index Training Data
    # vtt.load_train_data()
    # Split the URLs for easier checking and re-downloading.
    # vtt.split()
    # download_list = vtt.get_download_list()
    # for item in download_list:
    #     print("Downloading the {} list.".format(item))
    #     vtt.do_download_train(download_list[item])
    #     print("Finish downloading the {} list.".format(item))
    # vtt.indexer()

    # 3. Download Test Data
    vtt.load_test_data()
    vtt.do_download_test()

    # 4. Load and Match Category.
    # ...
    # vtt.load_category_data()
    # ...
    # category_name = vtt.get_category_name(category_id)
