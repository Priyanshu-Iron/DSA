// Find the Union of Two Sorted Arrays

// Brute Force Method
let arr1 = [1,2,3,4,5];
let arr2 = [1,2,3,4,6];

function FindUnion (arr1,arr2) {
    
    let mySet = new Set();

    for(let i=0; i<arr1.length; i++){
        mySet.add(arr1[i]);
    }

    for(let i=0; i<arr2.length; i++){
        mySet.add(arr2[i]);
    }
    
    let UnionArray = Array.from(mySet).sort((a, b) => a - b);

    console.log("Union of Two Sorted Arrays");
    console.log(UnionArray.join(' '));
    
}

// FindUnion(arr1,arr2);

function SortedUnionArray (arr1,arr2) {
    
    let i = 0;
    let j = 0;
    let UnionArray = [];

    while (i<arr1.length && j<arr2.length) {
        if (arr1[i] <= arr2[j]) {
            if (UnionArray.length === 0 || UnionArray[UnionArray.length-1] !== arr1[i]) {
                UnionArray.push(arr1[i]);
            }
            i++;
        } else{
            if (UnionArray.length === 0 || UnionArray[UnionArray.length-1] !== arr2[j]) {
                UnionArray.push(arr2[j]);
            }
            j++;
        }
    }
    
    while (i<arr1.length) {
        if (UnionArray.length === 0 || UnionArray[UnionArray.length-1] !== arr1[i]) {
            UnionArray.push(arr1[i]);
        }
        i++;
    }

    while(j<arr2.length){
        if (UnionArray.length === 0 || UnionArray[UnionArray.length-1] !== arr2[j]) {
            UnionArray.push(arr2[j]);
        }
        j++;
    }

    return UnionArray;
}

console.log(`Union of Two Sorted Array is ${SortedUnionArray(arr1,arr2)}`);