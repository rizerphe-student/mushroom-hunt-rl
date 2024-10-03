import numpy as np
import pygame


class MushroomVisualizer:
    def __init__(self, grid_size, cell_size=3):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mushroom Field")

        self.colors = {
            "background": (50, 50, 50),
            "mushroom": (200, 100, 50),
            "agent": (0, 255, 0),
        }

    def draw(self, grid, agent_positions):
        self.screen.fill(self.colors["background"])

        # Draw mushrooms
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.colors["mushroom"],
                        (
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )

        # Draw agents
        for agent_pos in agent_positions:
            pygame.draw.circle(
                self.screen,
                self.colors["agent"],
                (
                    int(agent_pos[1] * self.cell_size),
                    int(agent_pos[0] * self.cell_size),
                ),
                self.cell_size // 2,
            )

        pygame.display.flip()

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

    def close(self):
        pygame.quit()
