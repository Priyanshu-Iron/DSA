// Find the Union of Two Sorted Arrays

// Brute Force Method
let arr1 = [1,2,3,4,5];
let arr2 = [1,2,3,4];

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
    while (i<arr1.length && i<arr2.length) {
        if (arr1[i] <= arr2[j]) {
            if (UnionArray.back() != arr1[i]) {
                UnionArray.push_back(arr1[i]);
            }
            i++;
        }
    }
    return UnionArray;
}