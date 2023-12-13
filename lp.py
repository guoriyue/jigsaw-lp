from typing import List, Tuple, Dict
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


from scipy.spatial.distance import mahalanobis

MGC_NUM_ROTATIONS = {"T": 0, "R": 1, "D": 2, "L": 3}
def mgc(image1, image2, orientation):
    """
    Calculate the Mahalanobis Gradient Compatibility (MGC) of image 1 relative
    to image 2. MGC provides a measure of the similarity in gradient
    distributions between the boundaries of adjoining images with respect to
    a particular orientation. For detailed information on the underlying,
    please see Gallagher et al. (2012).

    Orientations are integers, defined according to Yu et al. (2015):
    - 0: measure MGC between the top of image 1 and bottom of image 2;
    - 1: measure MGC between the right of image 1 and left of image 2;
    - 2: measure MGC between the bottom of image 1 and top of image 2;
    - 3: measure MGC between the left of image 1 and right of image 2;

    Both images are first rotated into position according to the specified
    orientations, such that the right side of image 1 and the left side of
    image 2 are the boundaries of interest. This preprocessing step simplifies
    the subsequent calculation of the MGC, but increases computation time.
    Therefore, a straightforward optimisation would be to extract boundary
    sequences directly.

    NOTE: nomenclature taken from Gallagher et al. (2012).

    :param orientation: orientation image 1 relative to image 2.
    :param image1: first image.
    :param image2: second image.
    :return MGC.
    """
    # assert image1.shape == image2.shape, 'images must be of same dimensions'
    # assert orientation in MGC_NUM_ROTATIONS, 'invalid orientation'

    num_rotations = MGC_NUM_ROTATIONS[orientation]

    # Rotate images based on orientation - this is easier than extracting
    # the sequences based on an orientation case switch
    # print("image1.shape", image1.shape, image1)
    image1_signed = np.rot90(image1.image, num_rotations).astype(np.int16)
    image2_signed = np.rot90(image2.image, num_rotations).astype(np.int16)

    # Get mean gradient of image1

    g_i_l = image1_signed[:, -1] - image1_signed[:, -2]
    mu = g_i_l.mean(axis=0)

    # Get covariance matrix S
    # Small values are added to the diagonal of S to resolve non-invertibility
    # of S. This will not influence the final result.

    s = np.cov(g_i_l.T) + np.eye(3) * 10e-6

    # Get G_ij_LR

    g_ij_lr = image2_signed[:, 1] - image1_signed[:, -1]

    return sum(mahalanobis(row, mu, np.linalg.inv(s)) for row in g_ij_lr)


class Piece(object):
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from gaps.piece import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index):
        self.image = image[:]
        self.id = index

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape

def dissimilarity_measure(first_piece, second_piece, orientation):
    # def compute_mgc_distances(images, pairwise_matches):
    # """
    # Compute MGC distances for all specified images and their pairwise matches.

    # :param images: list of images.
    # :param pairwise_matches: list of (image index 1, image index 2, orientation)
    #  tuples.
    # :return: dictionary with tuples from pairwise_matches as keys, and their
    # resulting MGCs as values.
    # """
    # return {(i, j, o): mgc(images[i], images[j], o) for
    #         i, j, o in pairwise_matches}
    
    return mgc(first_piece, second_piece, orientation[0]), mgc(second_piece, first_piece, orientation[1])

    rows, columns, _ = first_piece.shape()
    color_difference = None


    # | L | - | R |
    if orientation == "LR":
        color_difference = (
            first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]
        )

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        color_difference = (
            first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]
        )

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)
    return value

def flatten_image(image, piece_size, indexed=False):
    """Converts image into list of square pieces.

    Input image is divided into square pieces of specified size and than
    flattened into list. Each list element is PIECE_SIZE x PIECE_SIZE x 3

    :params image:      Input image.
    :params piece_size: Size of single square piece.
    :params indexed: If True list of Pieces with IDs will be returned,
        otherwise list of ndarray pieces

    Usage::

        >>> from gaps.image_helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)

    """
    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = (
                x * piece_size,
                y * piece_size,
                (x + 1) * piece_size,
                (y + 1) * piece_size,
            )
            piece = np.empty((piece_size, piece_size, image.shape[2]))
            piece[:piece_size, :piece_size, :] = image[top:h, left:w, :]
            pieces.append(piece)

    if indexed:
        pieces = [Piece(value, index) for index, value in enumerate(pieces)]

    return pieces, rows, columns



class ImageAnalysis(object):
    """Cache for dissimilarity measures of individuals

    Class have static lookup table where keys are Piece's id's.  For each pair
    puzzle pieces there is a map with values representing dissimilarity measure
    between them. Each next generation have greater chance to use cached value
    instead of calculating measure again.

    Attributes:
        dissimilarity_measures: Dictionary with cached dissimilarity measures for pieces
        best_match_table: Dictionary with best matching piece for each edge and piece

    """

    dissimilarity_measures: Dict[Tuple, Dict[str, float]] = {}
    best_match_table: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    @classmethod
    def analyze_image(cls, pieces):
        for piece in pieces:
            # For each edge we keep best matches as a sorted list.
            # Edges with lower dissimilarity_measure have higher priority.
            cls.best_match_table[piece.id] = {"T": [], "R": [], "D": [], "L": []}

        def update_best_match_table(first_piece, second_piece):
            measure = dissimilarity_measure(first_piece, second_piece, orientation)
            # print("measure", measure)
            # cls.put_dissimilarity(
            #     (first_piece.id, second_piece.id), orientation, measure
            # )
            # cls.best_match_table[first_piece.id][orientation].append(
            #     (second_piece.id, measure)
            # )
                
            cls.best_match_table[second_piece.id][orientation[1]].append(
                (first_piece.id, measure[1])
            )
            
            cls.best_match_table[first_piece.id][orientation[0]].append(
                (second_piece.id, measure[0])
            )

        # Calculate dissimilarity measures and best matches for each piece.
        iterations = len(pieces) - 1
        for first in range(iterations):
            for second in range(first + 1, len(pieces)):
                # for orientation in ["L", "D", "R", "T"]:
                #     update_best_match_table(pieces[first], pieces[second])
                #     # update_best_match_table(pieces[second], pieces[first])
                for orientation in ["LR", "TD"]:
                    update_best_match_table(pieces[first], pieces[second])
                    update_best_match_table(pieces[second], pieces[first])

        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])

        return dict(sorted(cls.best_match_table.items()))

    # @classmethod
    # def put_dissimilarity(cls, ids, orientation, value):
    #     """Puts a new value in lookup table for given pieces

    #     :params ids:         Identfiers of puzzle pieces
    #     :params orientation: Orientation of puzzle pieces. Possible values are:
    #                          'LR' => 'Left-Right'
    #                          'TD' => 'Top-Down'
    #     :params value:       Value of dissimilarity measure

    #     Usage::

    #         >>> from gaps.image_analysis import ImageAnalysis
    #         >>> ImageAnalysis.put_dissimilarity([1, 2], "TD", 42)
    #     """
    #     if ids not in cls.dissimilarity_measures:
    #         cls.dissimilarity_measures[ids] = {}
    #     cls.dissimilarity_measures[ids][orientation] = value

    @classmethod
    def get_dissimilarity(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.get_dissimilarity([1, 2], "TD")

        """
        return cls.dissimilarity_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece, orientation):
        """ "Returns best match piece for given piece and orientation"""
        return cls.best_match_table[piece][orientation][0][0]


from operator import attrgetter

from gaps.individual import Individual
from scipy.optimize import linprog
from itertools import chain, groupby, product

# from ortools.linear_solver import pywraplp
# from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp
class LP(object):
    TERMINATION_THRESHOLD = 10

    def __init__(self, image, piece_size, population_size, generations, elite_size=2):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size
        pieces, self.rows, self.columns = flatten_image(image, piece_size, indexed=True)
        self._population = [
            Individual(pieces, self.rows, self.columns) for _ in range(population_size)
        ]
        self._pieces = pieces

    def start_evolution(self, maxiter=500000):
        print("=== Pieces:      {}\n".format(len(self._pieces)))

        best_match_table = ImageAnalysis.analyze_image(self._pieces)
        
        
        DELTA_X = [0, -1, 0, 1]
        DELTA_Y = [1, 0, -1, 0]
        
        NUM_ORIENTATIONS = 4
        # NUM_ROTATIONS = [3, 0, 1, 2]
        ROTATION_MAP = {"T": 2, "R": 3, "D": 0, "L": 1}
        
        
        
        # # solver = pywrapcp.Solver('Jigsaw')
        # # solver = cp_model.CpSolver()
        # # model = cp_model.CpModel()
        # # solver = pywraplp.Solver.CreateSolver("SAT")
        # # model = cp_model.CpModel()
        # model = pywrapcp.Solver("Jigsaw")


        # n = len(best_match_table)
        # # print("n", n)
        # # n is the total number of pieces
        # x_mat = [None for _ in range(n)]
        # y_mat = [None for _ in range(n)]
        # weight_ijo = [[{} for _ in range(n)] for _ in range(n)]
        # weights_x_sum = []
        # weights_y_sum = []
        # locations = []
        # for i in range(n):
        #     # locations.append(model.NewIntVar(0, n-1, "location_" + str(i)))
        #     locations.append(model.IntVar(0, n-1, "location_" + str(i)))
        
        # # model.AddAllDifferent(locations)
        # model.Add(model.AllDifferent(locations))
        #     # locations.append(solver.IntVar(0, n-1, "location_" + str(i)))
        # for i, first_piece in enumerate(best_match_table):
        #     # x_mat[first_piece] = model.NewIntVar(0, self.rows - 1, "x_" + str(first_piece))
        #     # y_mat[first_piece] = model.NewIntVar(0, self.columns - 1, "y_" + str(first_piece))
        #     # x_mat[first_piece] = solver.IntVar(0, self.rows - 1, "x_" + str(first_piece))
        #     # y_mat[first_piece] = solver.IntVar(0, self.columns - 1, "y_" + str(first_piece))
        #     for orientation in best_match_table[first_piece]:  
        #         # weight_ijo[first_piece][first_piece][orientation] = 1e7         
        #         for j, second_piece in enumerate(best_match_table[first_piece][orientation]):
        #             weight_ijo[first_piece][second_piece[0]][orientation] = second_piece[1]
        
        # print("x_mat", x_mat)
        # print("y_mat", y_mat)
        # sigma_o_x = {"T": 0, "R": -1, "D": 0, "L": 1}
        # sigma_o_y = {"T": self.columns, "R": 0, "D": -self.columns, "L": 0}
        # print("self.rows", self.rows)
        # print("self.columns", self.columns)
        # abs_diff_x_w = [[None for _ in range(n)] for _ in range(n)]
        # abs_diff_y_w = [[None for _ in range(n)] for _ in range(n)]
        # abs_diff_x = [[None for _ in range(n)] for _ in range(n)]
        # abs_diff_y = [[None for _ in range(n)] for _ in range(n)]
        # for i in range(n):
        #     for j in range(n):
        #         if not i == j:
        #             for o in ['T', 'R', 'D', 'L']:
        #                 # abs_diff_x_w[i][j] = model.NewIntVar(-self.rows + 1, self.rows - 1, "abs_diff_x_" + str(i) + "_" + str(j) + "_" + str(o))
        #                 # abs_diff_y_w[i][j] = model.NewIntVar(-self.columns + 1, self.columns - 1, "abs_diff_y_" + str(i) + "_" + str(j) + "_" + str(o))
        #                 abs_diff_x_w[i][j] = model.IntVar(0, self.rows, "abs_diff_x_w_" + str(i) + "_" + str(j) + "_" + str(o))
        #                 abs_diff_y_w[i][j] = model.IntVar(0, self.columns, "abs_diff_y_w_" + str(i) + "_" + str(j) + "_" + str(o))
        #                 # weight_ijo = solver.FloarVar()
        #                 # weight_ijo.set(weight_ijo[i][j][o])
        #                 print("weight_ijo[i][j][o]", weight_ijo[i][j][o])
        #                 weights_x_sum.append(abs_diff_x_w[i][j] * int(weight_ijo[i][j][o]))
        #                 weights_y_sum.append(abs_diff_y_w[i][j] * int(weight_ijo[i][j][o]))
        #                 # weights_x_sum.append((x_mat[i] - x_mat[j] - sigma_o_x[o]) * weight_ijo[i][j][o])
        #                 # weights_y_sum.append((y_mat[i] - y_mat[j] - sigma_o_y[o]) * weight_ijo[i][j][o])
                    
        # # # divide = 1.0/self.rows
        # for i in range(n):
        #     for j in range(i+1, n):
        #         if not i == j:
        #             for o in ['T', 'R', 'D', 'L']:
        #                 # pass
        #                 # print("i, j, o", i, j, o)
        #                 # print(x_mat[i], x_mat[j], sigma_o_x[o])
        #                 # model.Add(abs_diff_x_w[i][j] >= locations[i] - locations[j] - sigma_o_x[o])
        #                 # model.Add(abs_diff_x_w[i][j] >= -locations[i] + locations[j] + sigma_o_x[o])
                        
        #                 # model.Add(abs_diff_y_w[i][j] >= locations[i] - locations[j] - sigma_o_y[o])
        #                 # model.Add(abs_diff_y_w[i][j] >= -locations[i] + locations[j] + sigma_o_y[o])
        #                 pass
        #                 # # solver.Add(abs_diff_x_w[i][j] >= 1)
        #                 # model.Add(abs_diff_y_w[i][j] >= y_mat[i] - y_mat[j] - sigma_o_y[o])
        #                 # model.Add(abs_diff_y_w[i][j] >= -y_mat[i] + y_mat[j] + sigma_o_y[o])
                        
        #                 # model.Add(abs_diff_x_w[i][j] >= x_mat[i] - x_mat[j] - sigma_o_x[o])
        #                 # model.Add(abs_diff_x_w[i][j] >= -x_mat[i] + x_mat[j] + sigma_o_x[o])
        #                 # # solver.Add(abs_diff_x_w[i][j] >= 1)
        #                 # model.Add(abs_diff_y_w[i][j] >= y_mat[i] - y_mat[j] - sigma_o_y[o])
        #                 # model.Add(abs_diff_y_w[i][j] >= -y_mat[i] + y_mat[j] + sigma_o_y[o])
        #                 # solver.Add(abs_diff_x_w[i][j] >= x_mat[i] - x_mat[j] - sigma_o_x[o])
        #                 # solver.Add(abs_diff_x_w[i][j] >= -x_mat[i] + x_mat[j] + sigma_o_x[o])
        #                 # # solver.Add(abs_diff_x_w[i][j] >= 1)
        #                 # solver.Add(abs_diff_y_w[i][j] >= y_mat[i] - y_mat[j] - sigma_o_y[o])
        #                 # solver.Add(abs_diff_y_w[i][j] >= -y_mat[i] + y_mat[j] + sigma_o_y[o])
        #         # print("i, j", i, j)
        #         # solver.Add((x_mat[i] == x_mat[j] and y_mat[i] == y_mat[j]) == 0)
        #         # model.Add((x_mat[i] == x_mat[j] and y_mat[i] == y_mat[j]) == 0)
        
        # db = model.Phase(locations, model.CHOOSE_FIRST_UNBOUND, model.ASSIGN_MIN_VALUE) #model.CHOOSE_FIRST_UNBOUND, model.ASSIGN_MIN_VALUE

        # # Iterates through the solutions, displaying each.
        # num_solutions = 0
        # model.NewSearch(db)
        # while model.NextSolution():
        #     print("Solution", num_solutions, '\n')
        #     # Displays the solution just computed.
        #     for location in locations:
        #         print(location.Value())
        #     num_solutions += 1
        # model.EndSearch()
    
        
        # # solver.Minimize(solver.Sum(weights_x_sum + weights_y_sum))
        # # status = solver.Solve()
        # # solver = cp_model.CpSolver()
        # # # solver.Minimize(solver.Sum(x_mat + y_mat))
        # # status = solver.Solve(model)


        # # print("Number of constraints: ", solver.Constraints())
        
        # # for i in range(n):
        # #     for j in range(n):
        # #         solver.Add(abs_diff_x[i][j] >= x_mat[i] - x_mat[j])
        # #         solver.Add(abs_diff_x[i][j] >= -x_mat[i] + x_mat[j])
        # #         solver.Add(abs_diff_y[i][j] >= y_mat[i] - y_mat[j])
        # #         solver.Add(abs_diff_y[i][j] >= -y_mat[i] + y_mat[j])
        
        
        
        # # results = solver.Solve()
        # # for x_i in x_mat:
        # #     print(x_i.solution_value())
        # # for y_i in y_mat:
        # #     print(y_i.solution_value())
        # # if status == pywraplp.Solver.OPTIMAL:
        # #     print("Solution:")
        # #     print("Objective value =", solver.Objective().Value())
        # #     for x_i in x_mat:
        # #         print(x_i.solution_value())
        # #     for y_i in y_mat:
        # #         print(y_i.solution_value())
        
        # # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # #     print(f"Maximum of objective function: {solver.ObjectiveValue()}\n")
        # #     for location in locations:
        # #         print(f"location = {solver.Value(location)}")
        # #     # for y_i in y_mat:
        # #     #     print(f"y = {solver.Value(y_i)}")
        # #     # print(f"x = {solver.Value(x)}")
        # #     # print(f"y = {solver.Value(y)}")
        #     # print(f"z = {solver.Value(z)}")
        #     # print("x =", x.solution_value())
        #     # print("y =", y.solution_value())
        # # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # #     print("optimal results")
        # #     for x_i in x_mat:
        # #         print(f"x = {solver.Value(x_i)}")
        # #         # print(x_i.solution_value())
        # #     for y_i in y_mat:
        # #         # print(y_i.solution_value())
        # #         print(f"y = {solver.Value(y_i)}")
        # exit(0)
        
        
        
        
        active_selection = []
        # init_weights = []
        for first_piece in best_match_table:
            for orientation in best_match_table[first_piece]:
                for second_piece in best_match_table[first_piece][orientation]:
                    active_selection.append((first_piece, second_piece[0], orientation))
                    # init_weights.append(second_piece[1])
                    break
        init_weights = {}
        for first_piece in best_match_table:
            for orientation in best_match_table[first_piece]:
                for second_piece in best_match_table[first_piece][orientation]:
                    init_weights[(first_piece, second_piece[0], orientation)] = second_piece[1]
        
        DELTA_X = [0, -1, 0, 1]
        DELTA_Y = [1, 0, -1, 0]
        
        NUM_ORIENTATIONS = 4
        # NUM_ROTATIONS = [3, 0, 1, 2]
        # ROTATION_MAP = {"T": 2, "R": 3, "D": 0, "L": 1}
        ROTATION_MAP = {"T": 0, "R": 1, "D": 2, "L": 3}
        # ROTATION_MAP = {"T": 3, "R": 2, "D": 1, "L": 0}
        NUM_CONSTRAINTS = 2
        MATCH_REJECTION_THRESHOLD = 1e-5
        def compute_solution(starting_set, starting_weights, piece_duplicated=None):
             
            # def row_index(i, o):
            #     """
            #     Return the row index of the start of the constraints section for piece
            #     i and orientation o.

            #     :param i: the index of piece i.
            #     :param o: the orientation.
            #     :return: row index for piece i and orientation o.
            #     """
            #     return (NUM_ORIENTATIONS * NUM_CONSTRAINTS * i) + \
            #         (NUM_CONSTRAINTS * o)

            # Sort active selection by i and o. The resulting order allows for
            # simplifications on the inequality constraint matrix A_ub.
            
            

            # Number of rotations to align pieces for MGC calculation

            

            # print("len starting_set", len(starting_set))
            n = int(len(starting_set) / NUM_ORIENTATIONS)
            # print("starting_set", starting_set)
            # starting_set.sort(key=lambda x: x[0])
            # starting_set.sort(key=lambda x: x[2])

            # Construct inequality constraints matrix A_ub, given as follows:
            #    A_ub = | H1 | 0  | X |
            #           | 0  | H2 | Y |,
            # and where X = Y and H1 = H2 (constraints are identical for X and Y).

            # Recall that the inequality constraints are given as:
            #
            #   h_ijo >= x_i  - x_j - delta_o^x
            #   h_ijo >= -x_i + x_j + delta_o^x
            #
            # -delta_o^x and delta_o^x are the only constants and will be assigned to
            # the upper bounds of the inequality constraints. Therefore, the
            # constraints above are rewritten as follows:
            #
            #  -x_i + x_j + h_ijo >= -delta_o^x
            #   x_i - x_j + h_ijo >=  delta_o^x
            #
            # Rewriting than greater-than-or-equal signs to smaller-than-or-equal gives:
            #
            #   x_i - x_j - h_ijo <=  delta_o^x
            #  -x_i + x_j - h_ijo <= -delta_o^x
            #
            # Given these constraints, submatrices H1 and H2 are composed of two -1's
            # in each column:

            h_base = np.array([-1] * NUM_CONSTRAINTS + [0] * (NUM_ORIENTATIONS * NUM_CONSTRAINTS * n - NUM_CONSTRAINTS))
            H = np.array([np.roll(h_base, k) for k in range(0, NUM_ORIENTATIONS * NUM_CONSTRAINTS * n, NUM_CONSTRAINTS)]).T
            # print("H", H.shape)
            # print("h_base", h_base.shape)
            # print("H.shape", H.shape)
            # 112
            # 7*8*2
            
            # Submatrix X is composed of all (1, -1) pairs (for x_i and -x_i, resp.),
            # and (-1, 1) pairs for all (-x_j, x_j) in the inequality constraints.
            # Because of the fact that the active selection is sorted, X_i takes the
            # following form:
            #
            # | 1  0 0 0 0 0 ... |
            # | -1 0 0 0 0 0 ... |
            # | 0  1 0 0 0 0 ... |
            # | 0 -1 0 0 0 0 ... |
            # | 0  0 0 0 0 0 ... |
            # | ................ |
            # | 0 0 0 0 0 0 0 1  |
            # | 0 0 0 0 0 0 0 -1 |

            xi_base = np.array([1, -1] * NUM_ORIENTATIONS + [0] * (NUM_ORIENTATIONS * NUM_CONSTRAINTS) * (n - 1))
            Xi = np.array([np.roll(xi_base, k) for k in range(0, NUM_ORIENTATIONS * NUM_CONSTRAINTS * n, NUM_CONSTRAINTS * NUM_ORIENTATIONS)]).T
            # print("Xi", Xi.shape)
            # print("xi_base", xi_base.shape)
            # X_j is built dynamically depending on the values of the x_j's for the
            # oriented pairs (i, j, o).

            Xj = np.zeros(Xi.shape, dtype=np.int32)
            print("Xj", Xj.shape)
            print(len(starting_set))
            for piece_item in starting_set:
                i = piece_item[0]
                j = piece_item[1]
                o = piece_item[2]
                o = ROTATION_MAP[o]
                r = (NUM_ORIENTATIONS * NUM_CONSTRAINTS * i) + (NUM_CONSTRAINTS * o)
                # if not x_result[i] == 0:
                #     Xj[r, i] = 1
                # else:
                    # print("r", i, r)
                    # print("i, j, o, r", i, j, o, r)
                Xj[r:r + 2, j] = [-1, 1]
            X = Xi + Xj

            # Construct A_ub by vertically and horizontally stacking its constituent
            # matrices. Although pre-allocating the matrix and copying values may be
            # more efficient, it makes for less readable code.
            # Construct inequality constraints matrix A_ub, given as follows:
            #    A_ub = | H1 | 0  | X |
            #           | 0  | H2 | Y |,
            h, w = H.shape
            Z_h = np.zeros((h, w), dtype=np.int32)
            Z_x = np.zeros((h, n), dtype=np.int32)
            A_ub = np.vstack([H, Z_h])
            A_ub = np.hstack([A_ub, np.vstack([Z_h, H])])
            A_ub = np.hstack([A_ub, np.vstack([X, Z_x])])
            A_ub = np.hstack([A_ub, np.vstack([Z_x, X])])

            # Construct the upper bounds vector b_ub for the inequality constraints
            # matrix A_ub. This vector is given as follows:
            #
            # b_ub = [delta_o^x, -delta_o^x, delta_o^x, -delta_o^x, ....] +
            # [delta_o^y, -delta_o^y, delta_o^y, -delta_o^y, ...]

            b_x = list(chain.from_iterable([[DELTA_X[ROTATION_MAP[piece_item[2]]], -DELTA_X[ROTATION_MAP[piece_item[2]]]]
                                            for piece_item in starting_set]))
            b_y = list(chain.from_iterable([[DELTA_Y[ROTATION_MAP[piece_item[2]]], -DELTA_Y[ROTATION_MAP[piece_item[2]]]]
                                            for piece_item in starting_set]))
            b_ub = np.array(b_x + b_y)
            
            
            
            # for i in range(n):
            #     for j in range(i + 1, n):
            #         A_ub = np.vstack([A_ub, np.zeros(A_ub.shape[1])])
            #         A_ub[-1, i] = 1
            #         A_ub[-1, j] = -1
            #         A_ub = np.vstack([A_ub, np.zeros(A_ub.shape[1])])
            #         A_ub[-1, n + i] = 1
            #         A_ub[-1, n + j] = -1
            #         b_ub = np.append(b_ub, 0)  # Adjust the upper bounds accordingly

            # ... (continue with the rest of your code)

    # solution = linprog(c, A_ub, b_ub, options=options)


            # Construct coefficients vector c, to be used in the objective function. It
            # is given as:
            #
            # c = [w_ijo for all (i, j, o) in active selection]

            # c_base = [weights[_] for _ in starting_set]
            c_base = [starting_weights[piece_item] for piece_item in starting_set]
            # print("c_base", c_base)
            c = np.array(c_base * NUM_CONSTRAINTS + ([0] * NUM_CONSTRAINTS * n))

            # Calculate solution

            options = {'maxiter': maxiter} if maxiter else {}
            # print("SSSSSSShape")
            # print(A_ub.shape)
            # print(b_ub.shape)
            # print(c.shape)
            
            # Piece size: 128
            # === Pieces:      56
            
            print("A_ub", A_ub)
            print("b_ub", b_ub)
            print("c", c)
            solution = linprog(c, A_ub, b_ub, options=options)

            if not solution.success:
                if solution.message == 'Iteration limit reached.':
                    raise ValueError('iteration limit reached, try increasing the ' +
                                    'number of max iterations')
                else:
                    raise ValueError('unable to find solution to LP: {}'.format(
                        solution.message))

            xy = solution.x[-n * 2:]
            x, y = xy[:n], xy[n:]
            
            return x, y
        
        
        def compute_rejected_matches(active_selection, x, y):
            """
            Compute rejected matches given the active selection and the solution to the
            linear program given by the x and y coordinates of all pieces.

            For details, see Yu et al. (2015), equation 17.

            :param active_selection: list of (image index 1, image index 2,
            orientation) tuples representing the current active selection.
            :param x: x-coordinates of all pieces.
            :param y: y-coordinates of all pieces.
            :return: list of (image index 1, image index 2, orientation) tuples taken
            from the active selection, representing all rejected matches.
            """
            rejected_matches = set()
            for i, j, o in active_selection:
                if abs(x[i] - x[j] - DELTA_X[ROTATION_MAP[o]]) > MATCH_REJECTION_THRESHOLD:
                    rejected_matches.add((i, j, o))
                if abs(y[i] - y[j] - DELTA_Y[ROTATION_MAP[o]]) > MATCH_REJECTION_THRESHOLD:
                    rejected_matches.add((i, j, o))
            return rejected_matches
        
        def remove_rejected_matches(rejected_matches, best_match_table, init_weights):
            for i, j, o in rejected_matches:
                
                # for idx, tp in best_match_table[i][o]:
                #     if tp[0] == j:
                #         print("remove", tp)
                #         best_match_table[i][o].remove(tp)
                #         f+=1
                #         break
                o_rev = ""
                if o == "T":
                    o_rev = "D"
                elif o == "D":
                    o_rev = "T"
                elif o == "L":
                    o_rev = "R"
                elif o == "R":
                    o_rev = "L"
                
                # print("original length", len(best_match_table[i][o]), len(best_match_table[j][o_rev]))
                
                # for idx, tp in best_match_table[i][o]:
                #     if tp[0] == j:
                #         print("remove", tp)
                #         best_match_table[i][o].remove(tp)
                #         f+=1
                #         break
                best_match_table[i][o] = list(set(best_match_table[i][o]) - set([(j, init_weights[(i, j, o)])]))
                if len(best_match_table[i][o]) == 0:
                    print("ERROR 00000", i, j, o)
                    # best_match_table[i][o] = [(j, 1e3)]
                
                best_match_table[j][o_rev] = list(set(best_match_table[j][o_rev]) - set([(i, init_weights[(i, j, o)])]))
                if len(best_match_table[j][o_rev]) == 0:
                    print("ERROR 11111", i, j, o_rev)
                    # best_match_table[j][o_rev] = [(i, 1e3)]
                # print("new length", len(best_match_table[i][o]), len(best_match_table[j][o_rev]))
                # for tp in best_match_table[j][o_rev]:
                #     if tp[0] == i:
                #         print("remove", tp)
                #         best_match_table[j][o_rev].remove(tp)
                #         f+=1
                #         break
                # if not f==2:
                #     print("f", f)
                #     print("remove remove remove ERROR")

        
        def selected(first_piece, orientation, active_selection):
            for i, j, o in active_selection:
                if i == first_piece and o == orientation:
                    return True
            return False
    
        def count_selected(active_selection):
            pieces = set()
            for i, j, o in active_selection:
                pieces.add(i)
                pieces.add(j)
            return pieces
        
        x, y = compute_solution(active_selection, init_weights)
        # # print(x, y)
        # x_result = []
        # y_result = []
        # piece_result = []
        # matrix_fixed = np.zeros((self.rows, self.columns), dtype=np.int32)
        # matrix_fixed -= 1
        # piece_duplicated = []
        # for idx in range(len(x)):
        #     if not matrix_fixed[int(x[idx]), int(y[idx])] == -1:
        #         print("need to find a empty space")
        #         piece_duplicated.append(idx)
        #         piece_duplicated.append(matrix_fixed[int(x[idx]), int(y[idx])])
        #     else:
        #         matrix_fixed[int(x[idx]), int(y[idx])] = idx
        
        
        
        
        # for i in range(len(x)):
        #     if i in piece_duplicated:
        #         x_result.append(0)
        #         y_result.append(0)
        #     else:
        #         x_result.append(x[i])
        #         y_result.append(y[i])
        # x, y = x_result, y_result
        
        # # x, y = compute_solution(active_selection, init_weights, piece_duplicated)
        
        # while len(piece_result) < self.columns * self.rows:
            
        #     active_selection = []
        #     # init_weights = []
        #     for first_piece in best_match_table:
        #         if first_piece in piece_result:
        #             continue
        #         for orientation in best_match_table[first_piece]:
        #             for second_piece in best_match_table[first_piece][orientation]:
        #                 active_selection.append((first_piece, second_piece[0], orientation))
        #                 # init_weights.append(second_piece[1])
        #                 break
                    
        #     x, y = compute_solution(active_selection, init_weights)
            
        #     for idx in range(len(x)):
        #         if matrix_fixed[int(x[idx]), int(y[idx])] == 1:
        #             print("need to find a empty space")
        #         else:
        #             matrix_fixed[int(x[idx]), int(y[idx])] = 1
        #             x_result.append(x[idx])
        #             y_result.append(y[idx])
        #             piece_result.append(idx)
                
        # x, y = x_result, y_result
            
        # old_x, old_y = None, None
        # pieces = []
        # while (old_x is None and old_y is None) or not \
        #         (np.array_equal(old_x, x) and np.array_equal(old_y, y)):
        # # max_iter = 20
        # # while (len(set(x)) < self.rows - 1 or len(set(y)) < self.columns - 1) and max_iter:
        # #     max_iter -= 1
        #     # print(len(active_selection))
        #     # if not len(active_selection) == 960:
        #     #     print("active_selection", len(active_selection))
        #     #     break
            
        #     rejected_matches = compute_rejected_matches(active_selection, x, y)
        #     # print("rejected_matches", len(rejected_matches), rejected_matches)
        #     remove_rejected_matches(rejected_matches, best_match_table, init_weights)
                
        #     # print("rejected_matches", len(active_selection), len(rejected_matches), rejected_matches)
        #     # best_match_table = list(set(best_match_table) - rejected_matches)
        #     # print("pairwise_matches", len(best_match_table))
        #     # prev_active_selection = active_selection
        #     active_selection = []
        #     # init_weights = []
        #     for first_piece in best_match_table:
        #         for orientation in best_match_table[first_piece]:
        #             # if selected(first_piece, orientation, active_selection):
        #             #     continue
        #             for second_piece in best_match_table[first_piece][orientation]:
        #                 active_selection.append((first_piece, second_piece[0], orientation))
        #                 # init_weights.append(second_piece[1])
        #                 break
            
        #     # active_selection = compute_active_selection(active_selection,
        #     #                                             best_match_table)
        #     # print("active_selection", len(active_selection))
        #     old_x, old_y = x, y
        #     x, y = compute_solution(active_selection, init_weights)
        #     # return x, y
        #     # print(x, y)


        canvas = np.zeros((self.rows*self._piece_size, self.columns*self._piece_size, 3), dtype=np.uint8)

        xs = np.array(x, dtype=np.int32)
        ys = np.array(y, dtype=np.int32)
        
        for piece, (x, y) in zip(self._pieces, zip(xs, ys)):
            sx, sy, dx, dy = x * self._piece_size, y * self._piece_size, (x + 1) * self._piece_size, (y + 1) * self._piece_size
            canvas[sy:dy, sx:dx, :] = piece.image
        return canvas


    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))



import bisect

import cv2 as cv


class SizeDetector(object):
    """Detects piece size in pixels from given image

    Image is split into BGR single-channel images. Single-channel images are
    combined (R + G, R + B, G + B) in order to cover special edge cases where
    input image have one dominant color component.

    For each single channel-image size candidates are found and candidate with
    most occurrences is selected.

    :param image: Input puzzle with square pieces.

    Usage::

        >>> import cv
        >>> from gaps.size_detector import SizeDetector
        >>> image = cv.imread('puzzle.jpg')
        >>> detector = SizeDetector(image)
        >>> piece_size = detector.detect()

    """

    # Max absolute difference between width and height of bounding rectangle
    RECTANGLE_TOLERANCE = 3

    # Contour area / area of contours bounding rectangle
    EXTENT_RATIO = 0.75

    # Piece sizes bounds
    MIN_SIZE = 32
    MAX_SIZE = 128

    # Coefficient for MIN puzzle piece size
    MIN_SIZE_C = 0.9

    # Coefficient for MAX puzzle piece size
    MAX_SIZE_C = 1.3

    def __init__(self, image):
        self._image = image.copy()
        self._possible_sizes = []
        self._calculate_possible_sizes()

    def detect(self):
        """Detects piece size in pixels"""

        if len(self._possible_sizes) == 1:
            return self._possible_sizes[0]

        size_candidates = []
        for image in self._split_channel_images():
            candidates = self._find_size_candidates(image)
            size_candidates.extend(candidates)

        sizes_probability = {size: 0 for size in self._possible_sizes}
        for size_candidate in size_candidates:
            nearest_size = self._find_nearest_size(size_candidate)
            sizes_probability[nearest_size] += 1

        piece_size = max(sizes_probability, key=sizes_probability.get)
        return piece_size

    def _split_channel_images(self):
        blue, green, red = cv.split(self._image)

        split_channel_images = [
            red,
            green,
            blue,
            cv.add(red, green),
            cv.add(red, blue),
            cv.add(green, blue),
        ]

        return split_channel_images

    def _find_size_candidates(self, image):
        binary_image = self._filter_image(image)

        contours, _ = cv.findContours(
            binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        size_candidates = []
        for contour in contours:
            bounding_rect = cv.boundingRect(contour)
            contour_area = cv.contourArea(contour)
            if self._is_valid_contour(contour_area, bounding_rect):
                candidate = (bounding_rect[2] + bounding_rect[3]) / 2
                size_candidates.append(candidate)

        return size_candidates

    def _is_valid_contour(self, contour_area, bounding_rect):
        _, _, width, height = bounding_rect
        extent = float(contour_area) / (width * height)

        lower_limit = self.MIN_SIZE_C * self._possible_sizes[0]
        upper_limit = self.MAX_SIZE_C * self._possible_sizes[-1]

        is_valid_lower_range = width > lower_limit and height > lower_limit
        is_valid_upper_range = width < upper_limit and height < upper_limit
        is_square = abs(width - height) < self.RECTANGLE_TOLERANCE
        is_extent_valid = extent >= self.EXTENT_RATIO

        return (
            is_valid_lower_range
            and is_valid_upper_range
            and is_square
            and is_extent_valid
        )

    def _find_nearest_size(self, size_candidate):
        index = bisect.bisect_right(self._possible_sizes, size_candidate)

        if index == 0:
            return self._possible_sizes[0]

        if index >= len(self._possible_sizes):
            return self._possible_sizes[-1]

        right_size = self._possible_sizes[index]
        left_size = self._possible_sizes[index - 1]

        if abs(size_candidate - right_size) < abs(size_candidate - left_size):
            return right_size
        else:
            return left_size

    def _calculate_possible_sizes(self):
        """Calculates every possible piece size for given input image"""
        rows, columns, _ = self._image.shape

        for size in range(self.MIN_SIZE, self.MAX_SIZE + 1):
            if rows % size == 0 and columns % size == 0:
                self._possible_sizes.append(size)

    def _filter_image(self, image):
        _, thresh = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
        opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, (5, 5), iterations=3)

        return cv.bitwise_not(opened)


import click
import cv2 as cv
import numpy as np

puzzle = "puzzle.jpg"

size = None
population = 10
generations = 100
solution = "solution_lp.jpg"
debug = False

input_puzzle = cv.imread(puzzle)

if size is None:
    detector = SizeDetector(input_puzzle)
    size = detector.detect()
size = 128
click.echo(f"Population: {population}")
click.echo(f"Generations: {generations}")
click.echo(f"Piece size: {size}")

ga = LP(
    image=input_puzzle,
    piece_size=size,
    population_size=population,
    generations=generations,
)
result = ga.start_evolution()

# output_image = result.to_image()

cv.imwrite(solution, result)

# click.echo("Puzzle solved")