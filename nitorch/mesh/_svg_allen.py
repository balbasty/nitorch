

def parse_svg_allen(root):
    """Parse an SVG file of manual labels from the Allen brain atlas

    /!\ This is a very specific parser that does not work on most SVGss

    Parameters
    ----------
    root : str or xml.etree.ElementTree
        Filename (or root of XML tree) of an SVG file

    Returns
    -------
    structures : list[dict]
        Each element has format
            {'name': str, 'order': int, 'paths': list[dict]}
        Each 'path' element has format
            {'label': int, 'order': int,
             'path': list[float], 'subpaths': list[list[float]]}

    """
    if isinstance(root, str):
        import xml.etree.ElementTree as ET
        root = ET.parse(root).getroot()

    width = int(root.attrib.get('width', 0))
    height = int(root.attrib.get('height', 0))
    root = list(root)[0]
    scale = root.attrib.get('transform', None)
    if scale:
        scale = scale.strip()[6:-1].split(',')
        scale = [float(s.strip()) for s in scale]
        if len(scale) == 1:
            scale = scale * 2
    else:
        scale = 1, 1

    exclude = ('sulci', 'hotspot')
    structures = []
    for structure in root:
        structure_name = structure.attrib.get('graphic_group_label', '')
        if any(exc in structure_name.lower() for exc in exclude):
            continue
        order = int(structure.attrib.get('order', 0))
        structure_obj = {'name': structure_name, 'order': order}

        scale1 = structure.attrib.get('transform', None)
        if scale1:
            scale1 = scale1.strip()[6:-1].split(',')
            scale1 = [float(s.strip()) for s in scale1]
            if len(scale1) == 1:
                scale1 = scale1 * 2
            scale1 = scale1[0] * scale[0], scale1[1] * scale[1]
        else:
            scale1 = scale

        splines = []
        for path in structure:
            label = int(path.attrib.get('structure_id', 0))
            order = int(path.attrib.get('order', 0))
            spline_obj = {'order': order, 'label': label}
            try:
                spline, buf = parse_svg_path(path.attrib['d'])
                spline = [(x * scale1[0], y * scale1[1]) for x, y in spline]
                spline_obj['path'] = spline
                spline_obj['subpaths'] = []
                while buf:
                    spline, buf = parse_svg_path(buf)
                    spline = [(x * scale1[0], y * scale1[1]) for x, y in spline]
                    spline_obj['subpaths'].append(spline)
                splines.append(spline_obj)
            except ValueError as e:
                print(e)
        structure_obj['paths'] = splines
        structures.append(structure_obj)

    return structures, [width, height]


def parse_svg_path(path):
    """
    Parse a multi-bezier path from an SVG
    Non cubic segments are transformed into cubic segments.
    """

    def parse_float(buf):
        x = ''
        if buf[0] == '-':
            x += buf[0]
            buf = buf[1:]
        while buf and buf[0] in '0123456789.':
            x += buf[0]
            buf = buf[1:]
        x = float(x)
        return x, buf.lstrip(' \n\t,')

    def parse_M(buf):
        if not buf.startswith("M"):
            raise ValueError("Expected M to start")
        buf = buf[1:].lstrip(' \n\t,')
        x, buf = parse_float(buf)
        y, buf = parse_float(buf)
        return (x, y), buf.lstrip(' \n\t,')

    def parse_c(p0, buf):
        if not buf.startswith("c"):
            raise ValueError("Expected c to start")
        buf = buf[1:].lstrip(' \n\t,')
        p1x, buf = parse_float(buf)
        p1y, buf = parse_float(buf)
        p2x, buf = parse_float(buf)
        p2y, buf = parse_float(buf)
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        p1x += p0[0]
        p2x += p0[0]
        p3x += p0[0]
        p1y += p0[1]
        p2y += p0[1]
        p3y += p0[1]
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_C(buf):
        if not buf.startswith("C"):
            raise ValueError("Expected C to start")
        buf = buf[1:].lstrip(' \n\t,')
        p1x, buf = parse_float(buf)
        p1y, buf = parse_float(buf)
        p2x, buf = parse_float(buf)
        p2y, buf = parse_float(buf)
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_s(c0, p0, buf):
        if not buf.startswith("s"):
            raise ValueError("Expected s to start")
        buf = buf[1:].lstrip(' \n\t,')
        if c0:
            p1x = 2 * p0[0] - c0[0]
            p1y = 2 * p0[1] - c0[1]
        else:
            p1x, p1y = p0
        p2x, buf = parse_float(buf)
        p2y, buf = parse_float(buf)
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        p2x += p0[0]
        p3x += p0[0]
        p2y += p0[1]
        p3y += p0[1]
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_S(c0, p0, buf):
        if not buf.startswith("S"):
            raise ValueError("Expected S to start")
        buf = buf[1:].lstrip(' \n\t,')
        if c0:
            p1x = 2 * p0[0] - c0[0]
            p1y = 2 * p0[1] - c0[1]
        else:
            p1x, p1y = p0
        p2x, buf = parse_float(buf)
        p2y, buf = parse_float(buf)
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_l(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("l"):
            raise ValueError("Expected l to start")
        buf = buf[1:].lstrip(' \n\t,')
        p0x, p0y = p0
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        p3x += p0x
        p3y += p0y
        p1x, p1y = (1/3) * p0x + (2/3) * p3x, (1/3) * p0y + (2/3) * p3y
        p2x, p2y = (2/3) * p0x + (1/3) * p3x, (2/3) * p0y + (1/3) * p3y
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_v(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("v"):
            raise ValueError("Expected v to start")
        buf = buf[1:].lstrip(' \n\t,')
        p3y, buf = parse_float(buf)
        p3y += p0[1]
        p3 = (p0[0], p3y)
        return p0, p3, p3, buf.lstrip(' \n\t,')

    def parse_h(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("h"):
            raise ValueError("Expected h to start")
        buf = buf[1:].lstrip(' \n\t,')
        p3x, buf = parse_float(buf)
        p3x += p0[0]
        p3 = (p3x, p0[1])
        return p0, p3, p3, buf.lstrip(' \n\t,')

    def parse_L(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("L"):
            raise ValueError("Expected L to start")
        buf = buf[1:].lstrip(' \n\t,')
        p0x, p0y = p0
        p3x, buf = parse_float(buf)
        p3y, buf = parse_float(buf)
        p1x, p1y = (1/3) * p0x + (2/3) * p3x, (1/3) * p0y + (2/3) * p3y
        p2x, p2y = (2/3) * p0x + (1/3) * p3x, (2/3) * p0y + (1/3) * p3y
        return (p1x, p1y), (p2x, p2y), (p3x, p3y), buf.lstrip(' \n\t,')

    def parse_V(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("V"):
            raise ValueError("Expected V to start")
        buf = buf[1:].lstrip(' \n\t,')
        p3y, buf = parse_float(buf)
        p3 = (p0[0], p3y)
        return p0, p3, p3, buf.lstrip(' \n\t,')

    def parse_H(p0, buf):
        # I am cheating and transforming the linear segment into a cubic spline
        # (so that I can assume a single segment type later on)
        if not buf.startswith("H"):
            raise ValueError("Expected H to start")
        buf = buf[1:].lstrip(' \n\t,')
        p3x, buf = parse_float(buf)
        p3 = (p3x, p0[1])
        return p0, p3, p3, buf.lstrip(' \n\t,')

    buf = path.lstrip(' \n\t,')
    p, buf = parse_M(buf)
    points = [p]
    prev = ''
    while buf:
        cat = buf[0]
        if cat == 'c':
            p1, p2, p3, buf = parse_c(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'C':
            p1, p2, p3, buf = parse_C(buf)
            points.extend([p1, p2, p3])
        elif cat == 's':
            ctrl = points[-2] if prev in 'sScC' else None
            p1, p2, p3, buf = parse_s(ctrl, points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'S':
            ctrl = points[-2] if prev in 'sScC' else None
            p1, p2, p3, buf = parse_S(ctrl, points[-1], buf)
        elif cat == 'l':
            p1, p2, p3, buf = parse_l(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'L':
            p1, p2, p3, buf = parse_L(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'v':
            p1, p2, p3, buf = parse_v(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'V':
            p1, p2, p3, buf = parse_V(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'h':
            p1, p2, p3, buf = parse_h(points[-1], buf)
            points.extend([p1, p2, p3])
        elif cat == 'H':
            p1, p2, p3, buf = parse_H(points[-1], buf)
            points.extend([p1, p2, p3])
        else:
            break
        prev = cat

    if not buf or buf[0] != 'z':
        print(buf)
        raise ValueError('Expected z to end')
    else:
        buf = buf[1:].lstrip(' \n\t,')

    return points, buf