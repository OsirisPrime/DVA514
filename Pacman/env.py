# env.py
import numpy as np

# Försök importera era constants (UP, DOWN, LEFT, RIGHT, STOP etc.)
try:
    from constants import UP, DOWN, LEFT, RIGHT, STOP, TILEWIDTH, TILEHEIGHT
except Exception:
    # fallback om constants saknas vid import i vissa setup
    UP, DOWN, LEFT, RIGHT, STOP = 1, -1, 2, -2, 0
    TILEWIDTH, TILEHEIGHT = 16, 16


class PacmanEnv:
    """
    Minimal RL-wrapper runt ert Pacman-spel (pacmancode-stil).
    Kräver att game har update(dt) och att game.pacman finns.
    Pacman måste ha:
      - use_ai (bool)
      - ai_direction (int)
      - position (Vector2-liknande med x/y)
      - alive (bool)
    """

    def __init__(self, game, fixed_dt=1.0 / 60.0, max_steps=6000):
        self.game = game
        self.fixed_dt = float(fixed_dt)
        self.max_steps = int(max_steps)

        self.actions = [UP, DOWN, LEFT, RIGHT]  # 0..3
        self.n_actions = len(self.actions)

        # intern episod-state
        self.steps = 0
        self.prev_score = 0
        self.prev_pellet_count = None
        self.prev_alive = True
        self.prev_lives = self._get_lives()

        # Om du har power pellets / frightened: vi försöker detekta via score + events
        self.prev_level = None


        # Se till att pacman kan styras av AI
       # self._ensure_ai_control()

    def _boot_if_needed(self):
        """
    Ser till att game.pacman finns. Vissa pacmancode-versioner skapar pacman
    först när man kör restartGame/startGame/etc.
        """
        if hasattr(self.game, "pacman") and self.game.pacman is not None:
            return

        for fn in ["restartGame", "startGame", "startLevel", "start", "newGame", "newLevel"]:
            if hasattr(self.game, fn):
                getattr(self.game, fn)()
                if hasattr(self.game, "pacman") and self.game.pacman is not None:
                    return

        raise AttributeError("Pacman skapas inte av GameController (ingen game.pacman efter start/restart).")
    # --------- publikt API ---------
    def _get_lives(self):
        if hasattr(self.game, "lives"):
            return int(self.game.lives)
        return 0
    
    def reset(self):
        import pygame

        self.steps = 0

    # 1) starta/restarta spelet så pacman/ghosts/pellets skapas om
        if hasattr(self.game, "restartGame"):
            self.game.restartGame()
        elif hasattr(self.game, "startGame"):
            self.game.startGame()
        elif hasattr(self.game, "start"):
            self.game.start()

    # 2) säkerställ pacman + ai (skapar pacman om den inte finns)
        self._ensure_ai_control()

    # 3) släpp paus/READY
        if hasattr(self.game, "pause") and self.game.pause is not None:
            self.game.pause.paused = False
        if hasattr(self.game, "paused"):
            self.game.paused = False

    # 4) tvinga pacman till frisk start
        p = self.game.pacman
        if hasattr(p, "reset"):
            p.reset()
        if hasattr(p, "setPosition"):
            p.setPosition()

    # se till att AI har en start-riktning (så den inte ligger kvar i STOP)
        if not hasattr(p, "ai_direction"):
            p.ai_direction = STOP
        if getattr(p, "ai_direction", STOP) == STOP:
            p.ai_direction = LEFT

    # 5) kickstart: kör några ticks så att "READY"/spawn hinner släppa
    # (viktigt på mac: pumpa events så fönstret inte fryser)
        for _ in range(15):
            pygame.event.pump()
            if hasattr(self.game, "pause") and self.game.pause is not None:
                self.game.pause.paused = False
            if hasattr(self.game, "paused"):
                self.game.paused = False
            self.game.update()

    # 6) init “prev”
        self.prev_score = self._get_score()
        self.prev_pellet_count = self._get_pellet_count()
        self.prev_alive = self._is_alive()
        self.prev_level = getattr(self.game, "level", None)
        self.prev_lives = self._get_lives()

    # distans-minnen (om du har shaping)
        self._prev_pellet_dist = None
        self._prev_ghost_dist = None

        return self._get_state()

    def step(self, action_idx):
        """
        action_idx: 0..3 (UP/DOWN/LEFT/RIGHT)
        Kör spelet tills Pacman når nästa beslutspunkt (node/target) så att
        den inte vibrerar fram och tillbaka.
        """
        import pygame
        pygame.event.pump()
        self.steps += 1

    # --- släpp pause/READY varje step ---
        if hasattr(self.game, "pause") and self.game.pause is not None:
            self.game.pause.paused = False
        if hasattr(self.game, "paused"):
            self.game.paused = False

    # --- välj önskad riktning från action ---
        action_idx = int(action_idx)
        desired = self.actions[action_idx]
        current = getattr(self.game.pacman, "direction", STOP)

    # --- blockera direkt 180°-vändning (minskar oscillation) ---
        if hasattr(self.game.pacman, "oppositeDirection") and self.game.pacman.oppositeDirection(desired):
            desired = current

    # --- maska ogiltig riktning: välj bland giltiga ---
        if hasattr(self.game.pacman, "validDirection") and not self.game.pacman.validDirection(desired):
            valid = [d for d in self.actions if self.game.pacman.validDirection(d)]
            if valid:
    # Om current är STOP eller ogiltig → välj en giltig (inte current)
                if current == STOP or current not in valid:
                    desired = valid[np.random.randint(len(valid))]
                else:
        # annars håll current för stabilitet
                    desired = current

    # --- applicera beslutet ---
        self.game.pacman.ai_direction = desired

    # --- simulera fram tills nästa "decision point" (när pacman når target) ---
    # max ticks för att undvika att fastna om något går fel
        reached_decision = False
        for _ in range(30):
            if hasattr(self.game, "pause") and self.game.pause is not None:
                self.game.pause.paused = False

            self.game.update()

        # När pacman passerat target -> den kommer "snap:a" till node och kan byta riktning
            if hasattr(self.game.pacman, "overshotTarget") and self.game.pacman.overshotTarget():
                reached_decision = True
                break

        # Om den står still (STOP), bryt så vi inte loopar
            if getattr(self.game.pacman, "direction", STOP) == STOP:
                break

    # --- reward + done ---
        reward, info = self._compute_reward_info()
        lives = self._get_lives()
        alive = self._is_alive()

# Om pacman dog: ge straff, men avsluta INTE om liv kvar
        life_lost = (lives < getattr(self, "prev_lives", lives))
        self.prev_lives = lives

        done = False
        if lives <= 0:
            done = True
        elif self._level_cleared():
            done = True
        elif self.steps >= self.max_steps:
            done = True
        else:
            done = False
        #done = (not self._is_alive()) or self._level_cleared() or (self.steps >= self.max_steps)

    # Extra info för debug
        info["reached_decision"] = reached_decision
        info["pac_dir"] = int(getattr(self.game.pacman, "direction", STOP))

        state = self._get_state()
        return state, float(reward), bool(done), info

    def _get_state(self):
        """
        Feature-vector state (float32).
        Byggd för att vara enkel och snabb att träna på.
        """
        p = self._get_pacman_pos()

        # Normalisera med tile-size (så blir siffror i rimlig skala)
        px = p[0] / float(TILEWIDTH)
        py = p[1] / float(TILEHEIGHT)

        # riktning one-hot (UP,DOWN,LEFT,RIGHT,STOP)
        dir_oh = self._direction_onehot(getattr(self.game.pacman, "direction", STOP))

        # pellets: närmaste pellet-vektor och dist
        pellet_dx, pellet_dy, pellet_dist = self._nearest_pellet_features(p)

        # ghosts: närmaste ghost-vektor och dist + upp till 4 ghost positions
        ghost_feats = self._ghost_features(p)

        # steg-normalisering (valfritt, kan hjälpa)
        t = self.steps / float(self.max_steps)

        state = np.array(
            [px, py, t] +
            dir_oh +
            [pellet_dx, pellet_dy, pellet_dist] +
            ghost_feats,
            dtype=np.float32
        )
        return state

    def state_dim(self):
        """Praktisk helper om du vill veta dimensionen direkt."""
        s = self._get_state()
        return int(s.shape[0])

    # --------- reward ---------

    def _compute_reward_info(self):
        """
        Reward:
          +10 per pellet
          +1000 level clear
          -500 death
          -1 per step (för att undvika loopar)
          (liten shaping): +0.2 om närmare närmaste pellet, -0.3 om närmare ghost (risk)
        """
        info = {}

        score = self._get_score()
        alive = self._is_alive()

        pellet_count = self._get_pellet_count()
        if pellet_count is None:
            pellet_count = self.prev_pellet_count  # fallback

        reward = -1.0  # step penalty

        # pellet reward via pellet_count (bäst), annars via score-delta
        pellet_eaten = 0
        if self.prev_pellet_count is not None and pellet_count is not None:
            pellet_eaten = max(0, self.prev_pellet_count - pellet_count)
            if pellet_eaten > 0:
                reward += 10.0 * pellet_eaten

        # score delta (kan täcka power pellets/ghosts om ni har)
        score_delta = score - self.prev_score
        info["score_delta"] = int(score_delta)

        # Death
        if self.prev_alive and not alive:
            reward -= 500.0
            info["death"] = True
        else:
            info["death"] = False

        # Level clear
        if self._level_cleared():
            reward += 1000.0
            info["level_clear"] = True
        else:
            info["level_clear"] = False

        # Liten shaping (valfritt men ofta bra tidigt)
        # - belöna att komma närmare pellet
        # - straffa att komma närmare ghost
        p = self._get_pacman_pos()
        pellet_dist = self._nearest_pellet_distance(p)
        ghost_dist = self._nearest_ghost_distance(p)

        # Vi kan spara förra distanserna i info på ett enkelt sätt:
        prev_pd = getattr(self, "_prev_pellet_dist", None)
        prev_gd = getattr(self, "_prev_ghost_dist", None)

        if prev_pd is not None and pellet_dist is not None:
            if pellet_dist < prev_pd:
                reward += 0.2
            elif pellet_dist > prev_pd:
                reward -= 0.05

        if prev_gd is not None and ghost_dist is not None:
            if ghost_dist < prev_gd:
                reward -= 0.3  # blir närmare ghost = risk
            elif ghost_dist > prev_gd:
                reward += 0.05  # kommer längre bort = bra

        self._prev_pellet_dist = pellet_dist
        self._prev_ghost_dist = ghost_dist

        # uppdatera "prev"
        self.prev_score = score
        self.prev_pellet_count = pellet_count
        self.prev_alive = alive
        self.prev_level = getattr(self.game, "level", self.prev_level)

        info["pellet_eaten"] = int(pellet_eaten)
        info["score"] = int(score)
        info["steps"] = int(self.steps)
        return reward, info

    # --------- helpers: hitta objekt i er kodbas ---------

    def _ensure_ai_control(self):
        if not hasattr(self.game, "pacman") or self.game.pacman is None:
        # försök skapa pacman genom att starta spelet/level
            for fn in ["startGame", "startLevel", "start", "newGame", "newLevel"]:
                if hasattr(self.game, fn):
                    getattr(self.game, fn)()
                    break

        if not hasattr(self.game, "pacman") or self.game.pacman is None:
            raise AttributeError("game saknar .pacman (kan inte bygga env)")

        p = self.game.pacman
        if not hasattr(p, "ai_direction"):
             p.ai_direction = STOP
        if not hasattr(p, "use_ai"):
            p.use_ai = True
        p.use_ai = True

    def _get_score(self):
        # Vanligt: game.score
        if hasattr(self.game, "score"):
            return int(self.game.score)
        # Ibland: game.textgroup.score eller liknande – fallback till 0
        return 0

    def _is_alive(self):
        return bool(getattr(self.game.pacman, "alive", True))

    def _get_pacman_pos(self):
        pos = getattr(self.game.pacman, "position", None)
        if pos is None:
            return (0.0, 0.0)
        # pacmancode Vector2 brukar ha x/y
        return (float(pos.x), float(pos.y))

    def _direction_onehot(self, direction):
        # [UP,DOWN,LEFT,RIGHT,STOP]
        oh = [0.0, 0.0, 0.0, 0.0, 0.0]
        if direction == UP:
            oh[0] = 1.0
        elif direction == DOWN:
            oh[1] = 1.0
        elif direction == LEFT:
            oh[2] = 1.0
        elif direction == RIGHT:
            oh[3] = 1.0
        else:
            oh[4] = 1.0
        return oh

    def _get_pellet_list(self):
        # Vanliga namn: game.pellets (PelletGroup), game.pelletList, etc.
        candidates = [
            getattr(self.game, "pellets", None),
            getattr(self.game, "pelletList", None),
            getattr(self.game, "pellet_group", None),
        ]
        for c in candidates:
            if c is None:
                continue
            # PelletGroup i pacmancode brukar ha .pelletList
            if hasattr(c, "pelletList"):
                return list(c.pelletList)
            if isinstance(c, (list, tuple)):
                return list(c)
        return []

    def _get_pellet_count(self):
        pellets = self._get_pellet_list()
        return len(pellets) if pellets is not None else None

    def _get_ghost_list(self):
        # Vanliga namn: game.ghosts (GhostGroup), game.ghostgroup, game.ghosts.ghosts, etc.
        candidates = [
            getattr(self.game, "ghosts", None),
            getattr(self.game, "ghostgroup", None),
            getattr(self.game, "ghostGroup", None),
        ]
        for c in candidates:
            if c is None:
                continue
            # pacmancode GhostGroup brukar ha .ghosts (dict/list)
            if hasattr(c, "ghosts"):
                g = c.ghosts
                # kan vara dict
                if isinstance(g, dict):
                    return list(g.values())
                if isinstance(g, (list, tuple)):
                    return list(g)
            if isinstance(c, (list, tuple)):
                return list(c)
        return []

    # --------- pellets/ghost features ---------

    def _nearest_pellet_distance(self, pac_pos):
        pellets = self._get_pellet_list()
        if not pellets:
            return None
        px, py = pac_pos
        best = None
        for pel in pellets:
            pos = getattr(pel, "position", None)
            if pos is None:
                continue
            dx = float(pos.x) - px
            dy = float(pos.y) - py
            d = (dx * dx + dy * dy) ** 0.5
            if best is None or d < best:
                best = d
        return best

    def _nearest_pellet_features(self, pac_pos):
        pellets = self._get_pellet_list()
        if not pellets:
            # inga pellets kvar → neutral
            return 0.0, 0.0, 0.0

        px, py = pac_pos
        best_d = None
        best_dx, best_dy = 0.0, 0.0

        for pel in pellets:
            pos = getattr(pel, "position", None)
            if pos is None:
                continue
            dx = float(pos.x) - px
            dy = float(pos.y) - py
            d = (dx * dx + dy * dy) ** 0.5
            if best_d is None or d < best_d:
                best_d = d
                best_dx, best_dy = dx, dy

        # normalisera
        best_dx /= float(TILEWIDTH)
        best_dy /= float(TILEHEIGHT)
        best_dist = 0.0 if best_d is None else (best_d / float(TILEWIDTH))
        return float(best_dx), float(best_dy), float(best_dist)

    def _nearest_ghost_distance(self, pac_pos):
        ghosts = self._get_ghost_list()
        if not ghosts:
            return None
        px, py = pac_pos
        best = None
        for gh in ghosts:
            pos = getattr(gh, "position", None)
            if pos is None:
                continue
            dx = float(pos.x) - px
            dy = float(pos.y) - py
            d = (dx * dx + dy * dy) ** 0.5
            if best is None or d < best:
                best = d
        return best

    def _ghost_features(self, pac_pos, max_ghosts=4):
        ghosts = self._get_ghost_list()
        px, py = pac_pos

        # sortera ghosts efter avstånd så närmaste kommer först
        ghost_items = []
        for gh in ghosts:
            pos = getattr(gh, "position", None)
            if pos is None:
                continue
            dx = float(pos.x) - px
            dy = float(pos.y) - py
            d = (dx * dx + dy * dy) ** 0.5
            ghost_items.append((d, dx, dy))

        ghost_items.sort(key=lambda t: t[0])

        # närmaste ghost vector + dist
        if ghost_items:
            d, dx, dy = ghost_items[0]
            ndx = dx / float(TILEWIDTH)
            ndy = dy / float(TILEHEIGHT)
            nd = d / float(TILEWIDTH)
        else:
            ndx = ndy = nd = 0.0

        feats = [float(ndx), float(ndy), float(nd)]

        # lägg på upp till 4 ghost positions (dx,dy) i närhetsordning
        # (om färre ghosts: padding 0)
        for i in range(max_ghosts):
            if i < len(ghost_items):
                _, dx, dy = ghost_items[i]
                feats.append(float(dx / float(TILEWIDTH)))
                feats.append(float(dy / float(TILEHEIGHT)))
            else:
                feats.append(0.0)
                feats.append(0.0)

        return feats

    # --------- done helpers ---------

    def _level_cleared(self):
        # enklast: inga pellets kvar
        pc = self._get_pellet_count()
        if pc is not None and pc == 0:
            return True
        return False