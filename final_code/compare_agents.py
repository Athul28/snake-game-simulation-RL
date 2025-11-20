import matplotlib.pyplot as plt
from agent import Agent
from student_agent import StudentAgent
from game import SnakeGameAI

def evaluate(agent_class, games=20):
    agent = agent_class()
    game = SnakeGameAI()

    scores = []

    for _ in range(games):
        score = 0
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, done, s = game.play_step(action)
            score = s

            if done:
                scores.append(score)
                game.reset()
                break

    avg_score = sum(scores) / games
    return avg_score, scores


if __name__ == "__main__":
    print("Evaluating Teacher Agent...")
    avg_teacher, teacher_scores = evaluate(Agent)
    print("Teacher Avg Score:", avg_teacher)

    print("\nEvaluating Student Agent...")
    avg_student, student_scores = evaluate(StudentAgent)
    print("Student Avg Score:", avg_student)

    # Plot comparison
    plt.figure(figsize=(10,5))
    plt.plot(teacher_scores, marker='o', label='Teacher Scores')
    plt.plot(student_scores, marker='x', label='Student Scores')
    plt.title("Teacher vs Student Performance Comparison")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()