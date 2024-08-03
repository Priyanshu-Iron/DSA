let arr = [4,2,1,56,6]

function SelectionSort(arr,n){
    for(let i=0; i<=n-2; i++){
        let minIndex = select(arr,i);
        
        let temp = arr[minIndex];
        arr[minIndex]=arr[i];
        arr[i]=temp;
    }
}

function select(arr,i){
    let minIndex = i
    for(let j=i;j<arr.length;j++){
        if(arr[j]<arr[minIndex]) minIndex = j;
    }
    return minIndex;
}

SelectionSort(arr,arr.length)
console.log(arr);