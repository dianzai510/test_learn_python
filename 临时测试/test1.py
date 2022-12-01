import pathlib

result_path = pathlib.Path('results.txt')
# result_path.write_text('aaa')  # write results.txt
# result_path.write_text('ddd',newline=Tru
with result_path.open('w') as fp:
    fp.write('aaaaa\n')