const nums = [2, 2, 3, 1, 3, 2, 1, 1];

function FindMajorityNaive(arr) {
    const n = arr.length;
    const res = [];

    for (let i = 0; i < n; i++) {
        
        // Count the frequency of arr[i]
        let cnt = 0;
        for (let j = i; j < n; j++) {
            if (arr[j] === arr[i]) {
                cnt += 1;
            }
        }
      
        // Check if arr[i] is a majority element
        if (cnt > (n / 3)) {
          
            // Add arr[i] only if it is not already
            // present in the result
            if (res.length === 0 || arr[i] !== res[0]) {
                res.push(arr[i]);
            }
        }
      
        // If we have found two majority elements, 
        // we can stop our search
        if (res.length === 2) {
            if (res[0] > res[1]) {
                [res[0], res[1]] = [res[1], res[0]];
            }
            break;
        }
    }

    return res;
}

// const result = FindMajorityBF(nums);
// console.log(result.join(" "));

function FindMajorityBetter(arr) {
    const n = arr.length;
    const freq = {};
    const res = [];

    // find frequency of each number
    for (const ele of arr) {
        freq[ele] = (freq[ele] || 0) + 1;
    }

    // Iterate over each key-value pair in the hash map
    for (const it in freq) {
        const ele = Number(it);
        const cnt = freq[it];

        // Add the element to the result, if its frequency
        // is greater than floor(n/3)
        if (cnt > Math.floor(n / 3)) {
            res.push(ele);
        }
    }

    if (res.length === 2 && res[0] > res[1]) {
        [res[0], res[1]] = [res[1], res[0]];
    }
    return res;
}

// const result = FindMajorityBetter(nums)
// console.log(result.join(" "));

// Boyer-Moore’s Voting Algorithm 
function FindMajorityOptimal(nums) {
    if (!nums || nums.length === 0) return [];

    let candidate1 = null, candidate2 = null;
    let count1 = 0, count2 = 0;

    // Step 1: Find potential candidates
    for (let num of nums) {
        if (num === candidate1) {
            count1++;
        } else if (num === candidate2) {
            count2++;
        } else if (count1 === 0) {
            candidate1 = num;
            count1 = 1;
        } else if (count2 === 0) {
            candidate2 = num;
            count2 = 1;
        } else {
            count1--;
            count2--;
        }
    }

    // Step 2: Verify the candidates
    count1 = 0;
    count2 = 0;

    for (let num of nums) {
        if (num === candidate1) count1++;
        else if (num === candidate2) count2++;
    }

    let result = [];
    if (count1 > Math.floor(nums.length / 3)) result.push(candidate1);
    if (count2 > Math.floor(nums.length / 3) && candidate1 !== candidate2) result.push(candidate2);

    return result;
}

console.log(FindMajorityOptimal(nums));