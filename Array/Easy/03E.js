// Second Smallest Element in an Array

let arr = [2,4,2,2,1,4,5]

function secondSmallest(arr){
    let smallest = arr[0]
    let secondsmall = Math.max
    for(let i=1; i<arr.length; i++){
        if(arr[i]<smallest){
            secondsmall = smallest;
            smallest = arr[i];
        }
        else if(arr[i]!=smallest && arr[i]<secondsmall){
            secondsmall=arr[i];
        }
    }
    console.log(secondsmall);
}
secondSmallest(arr)