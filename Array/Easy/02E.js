// Second Largest Element in an Array

let arr = [25,3,7,8,3];

function secondLargest(arr){
    let largest = arr[0];
    let secondlarg = -1;
    for(let i=1; i<arr.length; i++){
        if (arr[i]>largest) {
            secondlarg = largest;
            largest= arr[i];
        }
        else if(arr[i]<largest && arr[i]>secondlarg){
            secondlarg = arr[i]
        }
    };
    console.log(secondlarg);
    
};
secondLargest(arr)