import re


def extract_srt(start_time, end_time, file_path):
    result = []
    subtitle_buf = {}
    if file_path.split('.')[-1] == 'srt':
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data_buf = file.readlines()
            is_multi_line = False  # This is for combining two consecutive non-blank lines in a single subtitle.
            count_empty_line = 0  # This is for recognizing the consecutive empty lines in the end of the file.
            line_buf = {}
            for line in data_buf:
                if re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                    line = line.strip()
                    trim = re.match(r'^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', line)
                    line_buf['timeclip'] = [
                        int(trim.group(1)) * 3600000 + int(trim.group(2)) * 60000 + int(trim.group(3)) * 1000 + int(
                            trim.group(4)),
                        int(trim.group(5)) * 3600000 + int(trim.group(6)) * 60000 + int(trim.group(7)) * 1000 + int(
                            trim.group(8))
                    ]
                elif re.match(r'^\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}', line):
                    line = line.strip()
                    trim = re.match(r'^(\d{2}):(\d{2}):(\d{2}).(\d{3}) --> (\d{2}):(\d{2}):(\d{2}).(\d{3})', line)
                    line_buf['timeclip'] = [
                        int(trim.group(1)) * 3600000 + int(trim.group(2)) * 60000 + int(trim.group(3)) * 1000 + int(
                            trim.group(4)),
                        int(trim.group(5)) * 3600000 + int(trim.group(6)) * 60000 + int(trim.group(7)) * 1000 + int(
                            trim.group(8))
                    ]
                # Every time an index is matched, we will consider as a new subtitle begins.
                elif re.match(r'^\d+$', line):
                    line_buf = {}
                    index = int(line.strip('\n'))
                    is_multi_line = False
                    count_empty_line = 0
                # Every time an '\n' is matched, we consider that this subtitle ends.
                elif re.match("\n", line):
                    is_multi_line = False
                    if count_empty_line == 0:
                        subtitle_buf[index] = line_buf
                    else:
                        count_empty_line += 1
                else:
                    trim = re.match(r'(\([\s\S]*:\))([\s\S]*)', line)
                    if trim:
                        line = trim.group(2).strip('-')
                    else:
                        line = line.strip('-')
                    if is_multi_line is False:
                        line_buf['content'] = line.strip()
                    else:
                        line_buf['content'] = line_buf['content'] + ' ' + line.strip()
                    is_multi_line = True
        for record in subtitle_buf:
            if subtitle_buf[record]['timeclip'][0] > start_time and subtitle_buf[record]['timeclip'][1] < end_time:
                result.append(subtitle_buf[record]['content'])
    else:
        print('Wrong file. Please input a .srt file.')
    return result


if __name__ == '__main__':
    file_path = '/home/zhz/PycharmProjects/Hysia/tests/test_DB/subtitles/BBT0624.srt'
    start_time = 200000
    end_time = 250000
    test_result = extract_srt(start_time, end_time, file_path)
    print(test_result)
