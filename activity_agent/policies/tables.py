"""Read policy catalog rows without mixing eligibility across different programs."""
from html.parser import HTMLParser
import re


class PolicyTables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.rows, self.tables = [], []
        self.cells, self.cell, self.spans = [], None, {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'table':
            self.depth += 1
            if self.depth == 1:
                self.rows, self.spans = [], {}
        if self.depth != 1:
            return
        if tag == 'tr':
            self.cells = []
        elif tag in ('td', 'th'):
            def span(name):
                try:
                    return max(1, min(30, int(attrs.get(name, 1))))
                except ValueError:
                    return 1
            self.cell = {'parts': [], 'rows': span('rowspan'), 'cols': span('colspan')}
        elif self.cell is not None and tag in ('br', 'p', 'li', 'div'):
            self.cell['parts'].append('\n')

    def handle_data(self, text):
        if self.depth and self.cell is not None:
            self.cell['parts'].append(text)

    def handle_endtag(self, tag):
        if self.depth == 1:
            if tag in ('td', 'th') and self.cell is not None:
                text = '\n'.join(' '.join(line.split()) for line in ''.join(self.cell['parts']).splitlines() if line.strip())
                self.cells.append((text, self.cell['rows'], self.cell['cols']))
                self.cell = None
            elif tag == 'tr':
                values = {col: text for col, (text, _) in self.spans.items()}
                carry = {col: (text, remaining - 1) for col, (text, remaining) in self.spans.items() if remaining > 1}
                col = 0
                for text, rows, cols in self.cells:
                    while col in values:
                        col += 1
                    for offset in range(cols):
                        values[col + offset] = text
                        if rows > 1:
                            carry[col + offset] = (text, rows - 1)
                    col += cols
                if values:
                    self.rows.append([values.get(i, '') for i in range(max(values) + 1)])
                self.spans = carry
            elif tag == 'table':
                self.tables.append(self.rows)
        if tag == 'table' and self.depth:
            self.depth -= 1


def policy_rows(html):
    parser = PolicyTables()
    parser.feed(html)
    result = []
    for table_index, rows in enumerate(parser.tables):
        headers = None
        for row_index, cells in enumerate(rows):
            compact = [re.sub(r'\s+', '', cell) for cell in cells]
            if '사업명' in compact and any('지원대상' in cell for cell in compact):
                headers = compact
                continue
            if headers is None or len(cells) != len(headers):
                continue
            values = dict(zip(headers, cells))
            target = next((v for k, v in values.items() if '지원대상' in k), '')
            application = next((v for k, v in values.items() if '신청방법' in k or '접수방법' in k), '')
            name = values.get('사업명', '')
            support = next((v for k, v in values.items() if '지원내용' in k or '사업내용' in k), '')
            agency = next((v for k, v in values.items() if '담당부서' in k or '담당기관' in k), '')
            if not name or not target or not application or name == '사업명':
                continue
            flat = lambda value: ' '.join(value.split())
            result.append({'record_id': f'table-{table_index}-row-{row_index}', 'name': name,
                'target': target, 'application': application, 'support': support, 'agency': agency,
                'text': f'사업명: {flat(name)}\n지원대상: {flat(target)}\n지원내용: {flat(support)}\n신청방법: {flat(application)}\n담당기관: {flat(agency)}',
                'source_cells': values})
            if len(result) >= 30:
                return result
    return result
