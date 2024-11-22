// node setMatrix0.js

var setZeroes = function(matrix) {
  let n = matrix.length;
  let m = matrix[0].length;
  
  let zeroFirstRow = false; // Flag to mark if the first row should be zeroed
  let zeroFirstCol = false; // Flag to mark if the first column should be zeroed

  // Check if the first row contains any zero
  for (let j = 0; j < m; j++) {
    if (matrix[0][j] === 0) {
      zeroFirstRow = true;
      break;
    }
  }

  // Check if the first column contains any zero
  for (let i = 0; i < n; i++) {
    if (matrix[i][0] === 0) {
      zeroFirstCol = true;
      break;
    }
  }

  // Use the first row and first column to mark zeros for other cells
  for (let i = 1; i < n; i++) {
    for (let j = 1; j < m; j++) {
      if (matrix[i][j] === 0) {
        matrix[i][0] = 0; // Mark the start of the row
        matrix[0][j] = 0; // Mark the start of the column
      }
    }
  }

  // Zero out cells based on markers in the first row and first column
  for (let i = 1; i < n; i++) {
    for (let j = 1; j < m; j++) {
      if (matrix[i][0] === 0 || matrix[0][j] === 0) {
        matrix[i][j] = 0;
      }
    }
  }

  // Zero the first row if needed
  if (zeroFirstRow) {
    for (let j = 0; j < m; j++) {
      matrix[0][j] = 0;
    }
  }

  // Zero the first column if needed
  if (zeroFirstCol) {
    for (let i = 0; i < n; i++) {
      matrix[i][0] = 0;
    }
  }

  return matrix;
};

let matrix = [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1]
];

const ans = setZeroes(matrix);

console.log("The Final matrix is: ");
for (let i = 0; i < ans.length; i++) {
  console.log(ans[i].join(" "));
}
