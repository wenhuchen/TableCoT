import csv
from typing import List

def generate_table_str(filname: str, limit: int = 0) -> str:
	with open(filname) as f:
		table = f.readlines()
		if limit > 0:
			table = ''.join(table[:limit])
		else:
			table = ''.join(table)
		table = table.replace('#', ' | ')
		return table


def generate_table_str2(filename: str, limit: int = 24) -> str:
	table_str = ''
	with open(filename) as f:
		csv_reader = csv.reader(f, delimiter=',', escapechar='\\')
		for line_no, line in enumerate(csv_reader):
			new_line = []
			for x in line:
				if len(x.split(' ')) > 10:
					x = ' '.join(x.split(' ')[:10])
					new_line.append(x.replace('\n', ' '))
				else:
					new_line.append(x.replace('\n', ' '))
			tmp = ' | '.join(new_line)
			tmp = tmp.replace('"', '')
			table_str += tmp + '\n'
			if line_no >= limit:
				break

	return table_str


def num_lines(filename: str) -> int:
	rows = 0
	with open(filename) as f:
		csv_reader = csv.reader(f, delimiter=',', escapechar='\\')
		for line_no, line in enumerate(csv_reader):
			rows += 1
	return rows


def get_first_k_rows(table_str: str, limit: int) -> str:
	rows = table_str.split('\n')[:limit]
	rows = '\n'.join(rows)
	rows += '\n......'
	return rows



def get_certain_columns(table_str: str, columns: list[str]) -> str:
	headers = table_str.split('\n')[0].split(' | ')

	remains = []
	remain_headers = []
	for i, header in enumerate(headers):
		if header in columns or i == 0:
			remains.append(i)
			remain_headers.append(header)
	sub_table_str = ' | '.join(remain_headers) + '\n'

	for line in table_str.split('\n')[1:]:
		cells = [cell for i, cell in enumerate(line.split(' | ')) if i in remains]
		content = ' | '.join(cells)
		sub_table_str += content + '\n'

	return sub_table_str