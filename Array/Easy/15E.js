// Finding Single Number in a Array.

// let nums = [2,2,1]

// function findSingleNumber(nums) {
//     let numberCount = {};

//     // Count occurrences of each number
//     for (let i = 0; i < nums.length; i++) {
//         if (numberCount[nums[i]] === undefined) {
//             numberCount[nums[i]] = 1;
//         } else {
//             numberCount[nums[i]] += 1;
//         }
//     }

//     // Find the number with a count of 1
//     for (let key in numberCount) {
//         if (numberCount[key] === 1) {
//             return parseInt(key);
//         }
//     }

//     return null; // In case there's no single occurrence number
// }

// console.log(findSingleNumber(nums)); 


// XOR operator

let nums = [2, 2, 1];

function findSingleNumber(nums) {
    let result = 0;

    // XOR all the numbers
    for (let i = 0; i < nums.length; i++) {
        result ^= nums[i];
    }

    return result;
}

console.log(findSingleNumber(nums)); // Output: 1
