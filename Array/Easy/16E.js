// Find the Number of Winning Players

/*
You are given an integer n representing the number of players in a game and a 2D array pick where pick[i] = [xi, yi] represents that the player xi picked a ball of color yi.

Player i wins the game if they pick strictly more than i balls of the same color. In other words,

Player 0 wins if they pick any ball.
Player 1 wins if they pick at least two balls of the same color.
...
Player i wins if they pick at least i + 1 balls of the same color.
Return the number of players who win the game.

Note that multiple players can win the game
*/

/*
    integer n = number of players in a game
    2D Array = pick[i] = [xi, yi]
    xi = player || yi = ball color
    Input: n = 4, pick = [[0,0],[1,0],[1,0],[2,1],[2,1],[2,0]]
    Output: 2
    Explanation:
    Player 0 and player 1 win the game, while players 2 and 3 do not win
*/

function NumberOfWinningPlayers(n, picks) {
    let playerBalls = Array(n).fill(null).map(() => ({})); // Array to store ball color counts for each player

    // Count the number of balls of each color picked by each player
    for (let [player, color] of picks) {
        if (!playerBalls[player][color]) {
            playerBalls[player][color] = 0;
        }
        playerBalls[player][color] += 1;
    }

    let winners = 0;

    // Check each player to see if they win
    for (let i = 0; i < n; i++) {
        for (let color in playerBalls[i]) {
            if (playerBalls[i][color] >= i + 1) {
                winners++;
                break; // Player wins, no need to check other colors
            }
        }
    }

    return winners;
}

let n = 4;
let picks = [[0, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 0]];

console.log(NumberOfWinningPlayers(n, picks)); // Output: 2


// Optimize
// function NumberOfWinningPlayers(n, picks) {
//     let playerBalls = Array(n).fill(0).map(() => new Map()); // Use a Map for each player's ball color counts
//     let winners = 0;
//     let hasWon = Array(n).fill(false); // Track which players have already won

//     // Count the balls and check for winners in one pass
//     for (let [player, color] of picks) {
//         let colorCount = playerBalls[player].get(color) || 0;
//         playerBalls[player].set(color, colorCount + 1);

//         // Check if the player wins (and hasn't already won)
//         if (!hasWon[player] && playerBalls[player].get(color) >= player + 1) {
//             winners++;
//             hasWon[player] = true; // Mark player as a winner
//         }
//     }

//     return winners;
// }

// let n = 2;
// let picks = [[0, 8], [0, 3]];

// console.log(NumberOfWinningPlayers(n, picks)); // Output: 1
