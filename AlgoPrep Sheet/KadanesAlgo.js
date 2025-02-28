const nums = [-2,1,-3,4,-1,2,1,-5,4];

function maxSubarraySum(arr){
    let res = arr[0];

    for (let i = 0; i < arr.length; i++) {
        let currSum = 0;

        for (let j = i; j < arr.length; j++) {
            currSum = currSum + arr[j];

            res = Math.max(res,currSum)
        }
    }

    return res;
}

// console.log(maxSubarraySum(nums));


function kadanesAlgo(arr){
    let sum = 0;
    let maxSum = arr[0];

    for (let i = 0; i < arr.length; i++) {
        sum = sum + arr[i];
        maxSum = Math.max(maxSum,sum);

        if(sum < 0){
            sum = 0;
        }
    }

    return maxSum;
}

console.log(kadanesAlgo(nums));