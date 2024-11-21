function generatePascalsTriangle(n) {
    const triangle = [];
    
    for (let i = 0; i < n; i++) {
        const row = Array(i + 1).fill(1); // Initialize the row with 1s
        
        for (let j = 1; j < i; j++) { // Calculate intermediate values
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
        }
        
        triangle.push(row); // Add the row to the triangle
    }
    
    return triangle;
}

// Example usage
const n = 5;
const triangle = generatePascalsTriangle(n);

console.log(JSON.stringify(triangle)); // Convert to JSON string for the required format
