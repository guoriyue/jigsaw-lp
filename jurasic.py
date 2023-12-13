from collections import defaultdict
import fileinput
import math

from ortools.sat.python import cp_model


def read_sections(lines):
    section = []
    for line in lines:
        line = line.rstrip()
        if line == '':
            yield section
            section = []
        else:
            section.append(line)
    yield section


def edge_seq_to_code(char_seq):
    return ''.join(char_seq)


def tile_to_edge_tuple(tile):
    top = edge_seq_to_code(tile[0])
    right = edge_seq_to_code(r[-1] for r in tile)
    bottom = edge_seq_to_code(reversed(tile[-1]))
    left = edge_seq_to_code(r[0] for r in reversed(tile))

    return top, right, bottom, left


def flip(tile_tuple):
    top, right, bottom, left = tile_tuple
    return tuple(
        map(
            edge_seq_to_code,
            (reversed(top), reversed(left), reversed(bottom), reversed(right))
        )
    )


assert flip(('...', '..#', '.#.', '.##')) == ('...', '##.', '.#.', '#..')


def rotate(tile_tuple, times):
    sides = list(tile_tuple)
    for _ in range(times):
        sides = sides[1:] + [sides[0]]
    return tuple(sides)


assert rotate(('..', '.#', '#.', '##'), 0) == ('..', '.#', '#.', '##')
assert rotate(('..', '.#', '#.', '##'), 1) == ('.#', '#.', '##', '..')
assert rotate(('..', '.#', '#.', '##'), 4) == ('..', '.#', '#.', '##')


def read_tiles(lines):
    for section in read_sections(lines):
        assert section[0].startswith('Tile'), section[0]
        tile_id = int(section[0].split(' ')[1][:-1])
        yield int(tile_id), tile_to_edge_tuple([list(s) for s in section[1:]])


def enumerate_tile_positions(tile):
    for flipped, tile in enumerate((tile, flip(tile))):
        for rot in range(4):
            yield flipped, rot, rotate(tile, rot)


tiles = list(read_tiles(fileinput.input()))

by_top_edge = defaultdict(list)
by_right_edge = defaultdict(list)
by_bottom_edge = defaultdict(list)
by_left_edge = defaultdict(list)


for tile_id, tile in tiles:
    for flipped, rot, tile in enumerate_tile_positions(tile):
        for coll, key in zip(
            (by_top_edge, by_right_edge, by_bottom_edge, by_left_edge),
            tile
        ):
            coll[key].append((tile_id, flipped, rot))


print(tiles)
print(by_top_edge['#....####.'])


model = cp_model.CpModel()

size = int(math.sqrt(len(tiles)))
print(size)

assignments = {}
assignments_by_xy = defaultdict(list)
assignments_by_tile = defaultdict(list)


def neighbor_cells_and_matches(x, y, tile):
    top, right, bottom, left = tile
    if y < size - 1: # top
        yield x, y + 1, by_bottom_edge[edge_seq_to_code(reversed(top))]
    if x < size - 1: # right
        yield x + 1, y, by_left_edge[edge_seq_to_code(reversed(right))]
    if y > 0:  # bottom
        yield x, y - 1, by_top_edge[edge_seq_to_code(reversed(bottom))]
    if x > 0:  # left
        yield x - 1, y, by_right_edge[edge_seq_to_code(reversed(left))]


for x in range(size):
    for y in range(size):
        for tile_id, tile in tiles:
            for flipped, rot, tile in enumerate_tile_positions(tile):
                assign = model.NewBoolVar(f'assign_{tile_id}_{flipped}_{rot}_{x}_{y}')
                assignments[(x, y, tile_id, flipped, rot)] = assign
                assignments_by_xy[(x, y)].append(assign)
                assignments_by_tile[tile_id].append(assign)


for x in range(size):
    for y in range(size):
        for tile_id, tile in tiles:
            for flipped, rot, tile in enumerate_tile_positions(tile):
                assign = assignments[(x, y, tile_id, flipped, rot)]
                for x_n, y_n, matching in neighbor_cells_and_matches(x, y, tile):
                    valid = []
                    for tile_n, flipped_n, rot_n in matching:
                        valid.append(assignments[x_n, y_n, tile_n, flipped_n, rot_n])
                    model.AddBoolOr(valid).OnlyEnforceIf(assign)


for assignment_dict in (assignments_by_xy, assignments_by_tile):
    for assignment_list in assignment_dict.values():
        model.Add(sum(assignment_list) == 1)


solver = cp_model.CpSolver()

print(solver.StatusName(solver.Solve(model)))

prod = 1
for v in assignments.values():
    if solver.Value(v):
        _, tile_id, flipped, rot, x, y = v.Name().split('_')
        print(x, y, tile_id, flipped, rot)
        if (int(x), int(y)) in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)):
            prod *= int(tile_id)

print(prod)
