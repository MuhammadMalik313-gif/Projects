"""
AI Chess Engine — Python (pygame)
Algorithms: Minimax · Alpha-Beta Pruning · Heuristic Evaluation · DFS
"""

import pygame
import sys
import copy
import time
import threading

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SQ = 72          # square size in pixels
BOARD_SIZE = SQ * 8
PANEL_W = 300
WIDTH = BOARD_SIZE + PANEL_W
HEIGHT = BOARD_SIZE

FPS = 60

# Colors
BG          = (13,  13,  13)
SURFACE     = (22,  22,  22)
BORDER      = (42,  42,  42)
ACCENT      = (232, 201, 109)
ACCENT2     = (109, 232, 180)
ACCENT3     = (109, 180, 232)
DANGER      = (232, 109, 109)
TEXT        = (232, 232, 232)
MUTED       = (102, 102, 102)
LIGHT_SQ    = (240, 217, 181)
DARK_SQ     = (181, 136,  99)
HIGHLIGHT   = (232, 201, 109, 120)
MOVE_HINT   = (109, 232, 180, 100)
LAST_SQ     = (232, 201, 109,  80)
CHECK_SQ    = (232,  80,  80, 160)
WHITE_COLOR = (255, 255, 255)

# Pieces
W, B = 'w', 'b'
PIECES = ['P','N','B','R','Q','K']

PIECE_VAL = {'P':100,'N':320,'B':330,'R':500,'Q':900,'K':20000}

# Piece-square tables (white POV, black mirrors row index)
PST = {
    'P': [[0,0,0,0,0,0,0,0],[50,50,50,50,50,50,50,50],[10,10,20,30,30,20,10,10],
          [5,5,10,25,25,10,5,5],[0,0,0,20,20,0,0,0],[5,-5,-10,0,0,-10,-5,5],
          [5,10,10,-20,-20,10,10,5],[0,0,0,0,0,0,0,0]],
    'N': [[-50,-40,-30,-30,-30,-30,-40,-50],[-40,-20,0,0,0,0,-20,-40],
          [-30,0,10,15,15,10,0,-30],[-30,5,15,20,20,15,5,-30],
          [-30,0,15,20,20,15,0,-30],[-30,5,10,15,15,10,5,-30],
          [-40,-20,0,5,5,0,-20,-40],[-50,-40,-30,-30,-30,-30,-40,-50]],
    'B': [[-20,-10,-10,-10,-10,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
          [-10,0,5,10,10,5,0,-10],[-10,5,5,10,10,5,5,-10],
          [-10,0,10,10,10,10,0,-10],[-10,10,10,10,10,10,10,-10],
          [-10,5,0,0,0,0,5,-10],[-20,-10,-10,-10,-10,-10,-10,-20]],
    'R': [[0,0,0,0,0,0,0,0],[5,10,10,10,10,10,10,5],[-5,0,0,0,0,0,0,-5],
          [-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],
          [-5,0,0,0,0,0,0,-5],[0,0,0,5,5,0,0,0]],
    'Q': [[-20,-10,-10,-5,-5,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
          [-10,0,5,5,5,5,0,-10],[-5,0,5,5,5,5,0,-5],
          [0,0,5,5,5,5,0,-5],[-10,5,5,5,5,5,0,-10],
          [-10,0,5,0,0,0,0,-10],[-20,-10,-10,-5,-5,-10,-10,-20]],
    'K': [[-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
          [-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
          [-20,-30,-30,-40,-40,-30,-30,-20],[-10,-20,-20,-20,-20,-20,-20,-10],
          [20,20,0,0,0,0,20,20],[20,30,10,0,0,10,30,20]],
}

FILES = 'abcdefgh'

# Unicode pieces
UNICODE = {
    ('w','P'):'♙',('w','N'):'♘',('w','B'):'♗',('w','R'):'♖',('w','Q'):'♕',('w','K'):'♔',
    ('b','P'):'♟',('b','N'):'♞',('b','B'):'♝',('b','R'):'♜',('b','Q'):'♛',('b','K'):'♚',
}


# ─────────────────────────────────────────────
#  BOARD STATE
# ─────────────────────────────────────────────
class Piece:
    __slots__ = ('side','type')
    def __init__(self, side, t):
        self.side = side
        self.type = t
    def copy(self):
        return Piece(self.side, self.type)


def init_board():
    b = [[None]*8 for _ in range(8)]
    back = ['R','N','B','Q','K','B','N','R']
    for c in range(8):
        b[0][c] = Piece('b', back[c])
        b[1][c] = Piece('b', 'P')
        b[6][c] = Piece('w', 'P')
        b[7][c] = Piece('w', back[c])
    return b


# ─────────────────────────────────────────────
#  MOVE GENERATION
# ─────────────────────────────────────────────
def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def opp(side):
    return 'b' if side == 'w' else 'w'

def pawn_moves(b, r, c, side, ep, moves):
    d = -1 if side == 'w' else 1
    start = 6 if side == 'w' else 1
    prom_row = 0 if side == 'w' else 7
    # single push
    if in_bounds(r+d, c) and b[r+d][c] is None:
        if r+d == prom_row:
            for pt in ('Q','R','B','N'):
                moves.append((r,c,r+d,c,'prom',pt))
        else:
            moves.append((r,c,r+d,c,'',''))
        # double push
        if r == start and b[r+2*d][c] is None:
            moves.append((r,c,r+2*d,c,'dp',''))
    # captures
    for dc in (-1,1):
        nr, nc = r+d, c+dc
        if not in_bounds(nr, nc): continue
        if b[nr][nc] and b[nr][nc].side != side:
            if nr == prom_row:
                for pt in ('Q','R','B','N'):
                    moves.append((r,c,nr,nc,'prom',pt))
            else:
                moves.append((r,c,nr,nc,'',''))
        if ep and (nr, nc) == ep:
            moves.append((r,c,nr,nc,'ep',''))

def knight_moves(b, r, c, side, moves):
    for dr,dc in ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)):
        nr,nc = r+dr, c+dc
        if in_bounds(nr,nc) and (b[nr][nc] is None or b[nr][nc].side != side):
            moves.append((r,c,nr,nc,'',''))

def slide_moves(b, r, c, side, moves, dirs):
    for dr,dc in dirs:
        nr,nc = r+dr, c+dc
        while in_bounds(nr,nc):
            if b[nr][nc]:
                if b[nr][nc].side != side:
                    moves.append((r,c,nr,nc,'',''))
                break
            moves.append((r,c,nr,nc,'',''))
            nr+=dr; nc+=dc

def king_moves(b, r, c, side, castling, moves, check_castling=True):
    for dr,dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        nr,nc = r+dr, c+dc
        if in_bounds(nr,nc) and (b[nr][nc] is None or b[nr][nc].side != side):
            moves.append((r,c,nr,nc,'',''))
    if not check_castling: return
    row = 7 if side == 'w' else 0
    if r == row and c == 4:
        k_key = 'wK' if side=='w' else 'bK'
        q_key = 'wQ' if side=='w' else 'bQ'
        if castling.get(k_key) and b[row][5] is None and b[row][6] is None:
            if not sq_attacked(b, row, 4, side, {}, None) and not sq_attacked(b, row, 5, side, {}, None):
                moves.append((r,c,row,6,'castle-k',''))
        if castling.get(q_key) and b[row][3] is None and b[row][2] is None and b[row][1] is None:
            if not sq_attacked(b, row, 4, side, {}, None) and not sq_attacked(b, row, 3, side, {}, None):
                moves.append((r,c,row,2,'castle-q',''))

def gen_pseudo(b, side, castling, ep):
    moves = []
    for r in range(8):
        for c in range(8):
            p = b[r][c]
            if not p or p.side != side: continue
            if p.type == 'P': pawn_moves(b,r,c,side,ep,moves)
            elif p.type == 'N': knight_moves(b,r,c,side,moves)
            elif p.type == 'B': slide_moves(b,r,c,side,moves,[(-1,-1),(-1,1),(1,-1),(1,1)])
            elif p.type == 'R': slide_moves(b,r,c,side,moves,[(-1,0),(1,0),(0,-1),(0,1)])
            elif p.type == 'Q': slide_moves(b,r,c,side,moves,[(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)])
            elif p.type == 'K': king_moves(b,r,c,side,castling,moves,check_castling=False)
    return moves

def sq_attacked(b, r, c, side, castling, ep):
    enemy = opp(side)
    for m in gen_pseudo(b, enemy, castling, ep):
        if m[2] == r and m[3] == c:
            return True
    return False

def find_king(b, side):
    for r in range(8):
        for c in range(8):
            p = b[r][c]
            if p and p.side == side and p.type == 'K':
                return (r, c)
    return None

def in_check(b, side, castling, ep):
    k = find_king(b, side)
    return k and sq_attacked(b, k[0], k[1], side, castling, ep)

def apply_move(b, castling, ep, move):
    fr,fc,tr,tc,flag,extra = move
    nb = [row[:] for row in b]
    # deep copy pieces
    for r in range(8):
        for c in range(8):
            if nb[r][c]: nb[r][c] = nb[r][c].copy()

    ncast = dict(castling)
    nep = None

    piece = nb[fr][fc]
    nb[tr][tc] = Piece(piece.side, extra) if flag == 'prom' else piece
    nb[fr][fc] = None

    if flag == 'dp':
        nep = ((fr+tr)//2, tc)
    if flag == 'ep':
        nb[fr][tc] = None
    if flag == 'castle-k':
        nb[tr][5] = nb[tr][7]; nb[tr][7] = None
    if flag == 'castle-q':
        nb[tr][3] = nb[tr][0]; nb[tr][0] = None

    if piece.type == 'K':
        if piece.side == 'w': ncast['wK']=False; ncast['wQ']=False
        else:                  ncast['bK']=False; ncast['bQ']=False
    if piece.type == 'R':
        if fr==7 and fc==7: ncast['wK']=False
        if fr==7 and fc==0: ncast['wQ']=False
        if fr==0 and fc==7: ncast['bK']=False
        if fr==0 and fc==0: ncast['bQ']=False

    return nb, ncast, nep

def gen_legal(b, side, castling, ep):
    pseudo = gen_pseudo(b, side, castling, ep)
    legal = []
    for m in pseudo:
        nb, nc, ne = apply_move(b, castling, ep, m)
        if not in_check(nb, side, nc, ne):
            legal.append(m)
    return legal


# ─────────────────────────────────────────────
#  HEURISTIC EVALUATION
# ─────────────────────────────────────────────
def evaluate(b):
    score = 0
    for r in range(8):
        for c in range(8):
            p = b[r][c]
            if not p: continue
            val = PIECE_VAL[p.type]
            pst_row = r if p.side == 'w' else 7 - r
            pst_val = PST[p.type][pst_row][c] if p.type in PST else 0
            score += (val + pst_val) if p.side == 'w' else -(val + pst_val)
    return score


# ─────────────────────────────────────────────
#  MINIMAX + ALPHA-BETA (DFS)
# ─────────────────────────────────────────────
stats = {'nodes': 0, 'pruned': 0}

def minimax(b, castling, ep, depth, alpha, beta, maximizing):
    stats['nodes'] += 1
    side = 'w' if maximizing else 'b'
    moves = gen_legal(b, side, castling, ep)

    if depth == 0 or not moves:
        if not moves:
            if in_check(b, side, castling, ep):
                return (None, 99999 - stats['nodes']*0.001 if not maximizing else -99999 + stats['nodes']*0.001)
            return (None, 0)  # stalemate
        return (None, evaluate(b))

    # Move ordering: captures first
    def cap_val(m):
        p = b[m[2]][m[3]]
        return PIECE_VAL[p.type] if p else 0

    moves.sort(key=cap_val, reverse=True)

    best_move = None
    if maximizing:
        best = float('-inf')
        for m in moves:
            nb, nc, ne = apply_move(b, castling, ep, m)
            _, score = minimax(nb, nc, ne, depth-1, alpha, beta, False)
            if score > best:
                best = score; best_move = m
            alpha = max(alpha, best)
            if beta <= alpha:
                stats['pruned'] += 1; break  # ← Alpha-Beta cut-off
        return best_move, best
    else:
        best = float('inf')
        for m in moves:
            nb, nc, ne = apply_move(b, castling, ep, m)
            _, score = minimax(nb, nc, ne, depth-1, alpha, beta, True)
            if score < best:
                best = score; best_move = m
            beta = min(beta, best)
            if beta <= alpha:
                stats['pruned'] += 1; break  # ← Alpha-Beta cut-off
        return best_move, best


# ─────────────────────────────────────────────
#  NOTATION
# ─────────────────────────────────────────────
def move_notation(b, move):
    fr,fc,tr,tc,flag,extra = move
    p = b[fr][fc]
    cap = 'x' if b[tr][tc] else ''
    pfx = FILES[fc] if p.type == 'P' and cap else ('' if p.type == 'P' else p.type)
    suf = '='+extra if flag == 'prom' else ''
    return f"{pfx}{cap}{FILES[tc]}{8-tr}{suf}"


# ─────────────────────────────────────────────
#  PYGAME GAME CLASS
# ─────────────────────────────────────────────
class ChessGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Chess Engine — Minimax · Alpha-Beta · Heuristic · DFS")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont('segoeuisymbol', 48)
        self.font_med = pygame.font.SysFont('segoeuisymbol', 18)
        self.font_sm  = pygame.font.SysFont('segoeuisymbol', 14)
        self.font_xs  = pygame.font.SysFont('segoeuisymbol', 12)

        self.new_game()

    def new_game(self):
        self.board    = init_board()
        self.castling = {'wK':True,'wQ':True,'bK':True,'bQ':True}
        self.ep       = None
        self.turn     = 'w'
        self.selected = None
        self.legal    = []
        self.last_from= None
        self.last_to  = None
        self.history  = []
        self.move_list= []
        self.game_over= False
        self.result   = None
        self.ai_mode  = True
        self.ai_depth = 3
        self.ai_thinking = False
        self.log_lines= [
            ("info",    "New game — Human (White) vs AI (Black)"),
            ("dfs",     f"DFS search depth: {self.ai_depth}"),
            ("heuristic","Eval: material + piece-square tables"),
        ]
        self.depth_slider = 3   # 1-4
        self.show_result  = False

    # ── MAIN LOOP ──
    def run(self):
        while True:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.on_click(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: self.new_game()
                if event.key == pygame.K_u: self.undo()
                if event.key == pygame.K_m: self.toggle_mode()
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.depth_slider = min(4, self.depth_slider+1); self.ai_depth = self.depth_slider
                if event.key == pygame.K_MINUS:
                    self.depth_slider = max(1, self.depth_slider-1); self.ai_depth = self.depth_slider

    def on_click(self, pos):
        x, y = pos
        # Board click
        if x < BOARD_SIZE and not self.game_over and not self.ai_thinking:
            if self.ai_mode and self.turn == 'b': return
            c, r = x // SQ, y // SQ
            if self.selected:
                move = next((m for m in self.legal if m[2]==r and m[3]==c), None)
                if move:
                    self.execute(move)
                    return
            p = self.board[r][c]
            if p and p.side == self.turn:
                self.selected = (r,c)
                self.legal = [m for m in gen_legal(self.board, self.turn, self.castling, self.ep)
                              if m[0]==r and m[1]==c]
            else:
                self.selected = None; self.legal = []
        # Panel buttons
        bx = BOARD_SIZE + 10
        # New Game button
        if bx <= x <= bx+120 and 20 <= y <= 46:
            self.new_game()
        # Undo
        if bx+130 <= x <= bx+210 and 20 <= y <= 46:
            self.undo()
        # Mode
        if bx <= x <= bx+140 and 56 <= y <= 82:
            self.toggle_mode()
        # Depth -
        if bx+170 <= x <= bx+190 and 56 <= y <= 82:
            self.depth_slider = max(1, self.depth_slider-1); self.ai_depth = self.depth_slider
        # Depth +
        if bx+210 <= x <= bx+230 and 56 <= y <= 82:
            self.depth_slider = min(4, self.depth_slider+1); self.ai_depth = self.depth_slider
        # Close result banner
        if self.show_result:
            rx = WIDTH//2-120; ry = HEIGHT//2-80
            if rx+80 <= x <= rx+160 and ry+100 <= y <= ry+130:
                self.new_game()

    def execute(self, move, is_ai=False):
        notation = move_notation(self.board, move)
        self.history.append((
            [row[:] for row in self.board], dict(self.castling),
            self.ep, self.turn, self.last_from, self.last_to, list(self.move_list)
        ))
        self.last_from = (move[0], move[1])
        self.last_to   = (move[2], move[3])
        self.board, self.castling, self.ep = apply_move(self.board, self.castling, self.ep, move)
        self.move_list.append(notation)
        self.turn = opp(self.turn)
        self.selected = None; self.legal = []
        self.check_game_over()
        if not self.game_over and self.ai_mode and self.turn == 'b':
            self.start_ai()

    def check_game_over(self):
        moves = gen_legal(self.board, self.turn, self.castling, self.ep)
        if not moves:
            self.game_over = True
            chk = in_check(self.board, self.turn, self.castling, self.ep)
            winner = 'Black' if self.turn == 'w' else 'White'
            self.result = (f"Checkmate! {winner} wins" if chk else "Stalemate — Draw")
            self.show_result = True

    def start_ai(self):
        self.ai_thinking = True
        self.log_lines.append(("dfs",       f"DFS tree — depth {self.ai_depth}"))
        self.log_lines.append(("minimax",    "Minimax: searching positions…"))
        t = threading.Thread(target=self.do_ai, daemon=True)
        t.start()

    def do_ai(self):
        stats['nodes'] = 0; stats['pruned'] = 0
        t0 = time.time()
        move, score = minimax(self.board, self.castling, self.ep,
                              self.ai_depth, float('-inf'), float('inf'), False)
        elapsed = int((time.time()-t0)*1000)
        self.log_lines.append(("alpha",     f"Alpha-Beta: pruned {stats['pruned']} branches"))
        self.log_lines.append(("heuristic", f"Eval score: {score:+}"))
        self.log_lines.append(("minimax",   f"Nodes: {stats['nodes']} | {elapsed}ms"))
        if move:
            note = move_notation(self.board, move)
            self.log_lines.append(("move", f"AI plays: {note}"))
            # Execute on main thread next draw cycle
            self._pending_ai_move = move
        self.ai_thinking = False

    def undo(self):
        if not self.history: return
        steps = 2 if (self.ai_mode and len(self.history)>=2) else 1
        for _ in range(steps):
            if self.history:
                b,cast,ep,turn,lf,lt,ml = self.history.pop()
                self.board=b; self.castling=cast; self.ep=ep
                self.turn=turn; self.last_from=lf; self.last_to=lt; self.move_list=ml
        self.game_over=False; self.show_result=False; self.selected=None; self.legal=[]
        self.log_lines.append(("info","Move undone"))

    def toggle_mode(self):
        self.ai_mode = not self.ai_mode
        label = "Human vs AI" if self.ai_mode else "Human vs Human"
        self.log_lines.append(("info", f"Mode: {label}"))

    # ── DRAWING ──
    def draw(self):
        # Handle pending AI move
        if hasattr(self,'_pending_ai_move') and self._pending_ai_move and not self.ai_thinking:
            m = self._pending_ai_move; self._pending_ai_move = None
            self.execute(m, True)

        self.screen.fill(BG)
        self.draw_board()
        self.draw_pieces()
        self.draw_panel()
        if self.show_result:
            self.draw_result()
        pygame.display.flip()

    def draw_board(self):
        for r in range(8):
            for c in range(8):
                color = LIGHT_SQ if (r+c)%2==0 else DARK_SQ
                rect = (c*SQ, r*SQ, SQ, SQ)
                pygame.draw.rect(self.screen, color, rect)

                # Last move highlight
                if self.last_from and (r,c)==self.last_from:
                    s = pygame.Surface((SQ,SQ), pygame.SRCALPHA); s.fill((*ACCENT, 70))
                    self.screen.blit(s, (c*SQ, r*SQ))
                if self.last_to and (r,c)==self.last_to:
                    s = pygame.Surface((SQ,SQ), pygame.SRCALPHA); s.fill((*ACCENT, 110))
                    self.screen.blit(s, (c*SQ, r*SQ))

                # Selected
                if self.selected and (r,c)==self.selected:
                    s = pygame.Surface((SQ,SQ), pygame.SRCALPHA); s.fill((*ACCENT, 140))
                    self.screen.blit(s, (c*SQ, r*SQ))
                    pygame.draw.rect(self.screen, ACCENT, rect, 3)

                # Legal move hints
                for m in self.legal:
                    if m[2]==r and m[3]==c:
                        if self.board[r][c]:
                            s = pygame.Surface((SQ,SQ), pygame.SRCALPHA); s.fill((*DANGER, 80))
                            self.screen.blit(s, (c*SQ, r*SQ))
                            pygame.draw.rect(self.screen, DANGER, rect, 2)
                        else:
                            s = pygame.Surface((SQ,SQ), pygame.SRCALPHA)
                            pygame.draw.circle(s, (*ACCENT2, 120), (SQ//2,SQ//2), 10)
                            self.screen.blit(s, (c*SQ, r*SQ))

                # In-check king
                p = self.board[r][c]
                if p and p.type=='K' and p.side==self.turn and in_check(self.board,self.turn,self.castling,self.ep):
                    s = pygame.Surface((SQ,SQ), pygame.SRCALPHA); s.fill((*DANGER, 140))
                    self.screen.blit(s, (c*SQ, r*SQ))

        # Coords
        for i in range(8):
            t = self.font_xs.render(str(8-i), True, (*MUTED,255))
            self.screen.blit(t, (3, i*SQ+4))
            t = self.font_xs.render(FILES[i], True, (*MUTED,255))
            self.screen.blit(t, (i*SQ+SQ-12, BOARD_SIZE-14))

        # Board border
        pygame.draw.rect(self.screen, BORDER, (0,0,BOARD_SIZE,BOARD_SIZE), 2)

    def draw_pieces(self):
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if not p: continue
                ch = UNICODE.get((p.side, p.type), '?')
                color = (248,240,220) if p.side=='w' else (30,30,30)
                # Shadow
                shadow = self.font_lg.render(ch, True, (0,0,0))
                self.screen.blit(shadow, (c*SQ+SQ//2-shadow.get_width()//2+2, r*SQ+SQ//2-shadow.get_height()//2+2))
                surf = self.font_lg.render(ch, True, color)
                self.screen.blit(surf, (c*SQ+SQ//2-surf.get_width()//2, r*SQ+SQ//2-surf.get_height()//2))

    def draw_panel(self):
        px = BOARD_SIZE
        pygame.draw.rect(self.screen, SURFACE, (px, 0, PANEL_W, HEIGHT))
        pygame.draw.line(self.screen, BORDER, (px,0),(px,HEIGHT), 2)
        x = px+12

        # ── Buttons ──
        self._btn(x,    20, 120, 26, "N  New Game", ACCENT)
        self._btn(x+130,20,  80, 26, "U  Undo",     TEXT)
        self._btn(x,    56, 140, 26,
                  "M  " + ("vs Human" if self.ai_mode else "vs AI"), ACCENT2)
        # Depth
        t = self.font_sm.render(f"Depth:", True, MUTED)
        self.screen.blit(t, (x+150, 62))
        self._btn(x+170,56,20,26,"−",TEXT)
        t2 = self.font_med.render(str(self.depth_slider), True, ACCENT)
        self.screen.blit(t2, (x+197, 60))
        self._btn(x+210,56,20,26,"+",TEXT)

        y = 100
        # ── Status ──
        self._section("STATUS", x, y)
        y += 22
        turn_col = TEXT if self.turn=='w' else ACCENT
        turn_str = ("♙ White" if self.turn=='w' else "♟ Black") + (" (AI)" if self.ai_mode and self.turn=='b' else "")
        self._label(f"Turn: {turn_str}", x, y, color=turn_col)
        y += 18

        mode_str = "Human vs AI" if self.ai_mode else "Human vs Human"
        self._label(f"Mode: {mode_str}", x, y, color=ACCENT2)
        y += 18

        if self.ai_thinking:
            dots = "·" * (int(time.time()*3)%4)
            self._label(f"AI thinking{dots}", x, y, color=ACCENT2)
        elif self.game_over:
            self._label(self.result or "Game Over", x, y, color=DANGER)
        y += 18

        # Score bar
        score = evaluate(self.board)
        pct = min(1.0, max(0.0, 0.5 + score/4000))
        bar_w = PANEL_W - 24
        pygame.draw.rect(self.screen, BORDER, (x, y, bar_w, 8), border_radius=2)
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            pygame.draw.rect(self.screen, ACCENT, (x, y, fill_w, 8), border_radius=2)
        self._label("Black", x, y+10, MUTED, 10)
        tw = self.font_xs.render("White", True, MUTED)
        self.screen.blit(tw, (x+bar_w-tw.get_width(), y+10))
        y += 30

        # ── Algorithm Log ──
        y += 4
        self._section("ALGORITHM LOG", x, y); y += 22
        max_lines = 12
        shown = self.log_lines[-max_lines:]
        col_map = {"minimax":ACCENT,"alpha":ACCENT2,"heuristic":ACCENT3,"dfs":DANGER,"move":TEXT,"info":MUTED}
        icons   = {"minimax":"⚡","alpha":"✂","heuristic":"📊","dfs":"🌲","move":"♟","info":"ℹ"}
        for kind, msg in shown:
            col = col_map.get(kind, TEXT)
            icon = icons.get(kind, "·")
            line = f"{icon} {msg}"
            t = self.font_xs.render(line[:38], True, col)
            self.screen.blit(t, (x, y)); y += 14
        y += 6

        # ── Move History ──
        self._section("MOVE HISTORY", x, y); y += 22
        moves_per_row = 2
        ml = self.move_list
        rows = (len(ml)+1)//2
        shown_rows = ml[-(min(rows,7)*2):]
        start_num = max(1, (len(ml)+1)//2 - 6)
        for i in range(0, len(shown_rows), 2):
            num = start_num + i//2
            t = self.font_xs.render(f"{num}.", True, MUTED)
            self.screen.blit(t, (x, y))
            if i < len(shown_rows):
                t2 = self.font_xs.render(shown_rows[i], True, TEXT)
                self.screen.blit(t2, (x+26, y))
            if i+1 < len(shown_rows):
                t3 = self.font_xs.render(shown_rows[i+1], True, ACCENT3)
                self.screen.blit(t3, (x+90, y))
            y += 14
        y += 6

        # ── Key hints ──
        hints = [("N","New Game"),("U","Undo"),("M","Toggle Mode"),("+/-","Depth")]
        self._section("KEYS", x, y); y+=22
        for k,v in hints:
            self._label(f"[{k}]  {v}", x, y, MUTED, 11); y+=14

    def draw_result(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,160))
        self.screen.blit(overlay, (0,0))
        bw, bh = 280, 150
        bx, by = WIDTH//2-bw//2, HEIGHT//2-bh//2
        pygame.draw.rect(self.screen, SURFACE, (bx,by,bw,bh), border_radius=4)
        pygame.draw.rect(self.screen, ACCENT, (bx,by,bw,bh), 2, border_radius=4)
        t = self.font_med.render(self.result or "Game Over", True, ACCENT)
        self.screen.blit(t, (bx+bw//2-t.get_width()//2, by+30))
        t2 = self.font_sm.render("Press N or click Play Again", True, MUTED)
        self.screen.blit(t2, (bx+bw//2-t2.get_width()//2, by+65))
        self._btn(bx+bw//2-40, by+100, 80, 28, "Play Again", ACCENT)

    def _btn(self, x, y, w, h, label, col):
        pygame.draw.rect(self.screen, SURFACE, (x,y,w,h), border_radius=2)
        pygame.draw.rect(self.screen, col, (x,y,w,h), 1, border_radius=2)
        t = self.font_xs.render(label, True, col)
        self.screen.blit(t, (x+w//2-t.get_width()//2, y+h//2-t.get_height()//2))

    def _section(self, title, x, y):
        t = self.font_xs.render(title, True, MUTED)
        self.screen.blit(t, (x, y))
        pygame.draw.line(self.screen, BORDER, (x, y+13), (BOARD_SIZE+PANEL_W-12, y+13))

    def _label(self, text, x, y, color=TEXT, size=12):
        f = self.font_xs if size<=12 else self.font_sm
        t = f.render(text, True, color)
        self.screen.blit(t, (x, y))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    game = ChessGame()
    game.run()
