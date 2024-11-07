// node LeaderInArray.js

/*
let arr = [16, 17, 4, 3, 5, 2];

function LeaderArray(arr) {
    let ans = [];
    for (let i = 0; i < arr.length; i++) {
        let leader = true;  // Assume current element is a leader
        for (let j = i + 1; j < arr.length; j++) {
            if (arr[j] > arr[i]) {  // If a larger element exists to the right
                leader = false;  // Current element is not a leader
                break;  // No need to check further elements to the right
            }
        }
        if (leader) {
            ans.push(arr[i]);  // Append the element to result if it's a leader
        }
    }
    return ans;
}

console.log(LeaderArray(arr));
*/


// optimal
function printLeaders(arr, n) {

    let ans = [];
  
    // Last element of an array is always a leader,
    // push into ans array.
    let max = arr[n - 1];
    ans.push(arr[n - 1]);
  
    // Start checking from the end whether a number is greater
    // than max no. from right, hence leader.
    for (let i = n - 2; i >= 0; i--) {
      if (arr[i] > max) {
        ans.push(arr[i]);
        max = arr[i];
      }
    }
  
    return ans;
  }
  
  // Array Initialization.
  let n = 6;
  let arr = [10, 22, 12, 3, 0, 6];
  
  let ans = printLeaders(arr, n);
  
  for (let i = ans.length - 1; i >= 0; i--) {
    console.log(ans[i]);
  }
  
  