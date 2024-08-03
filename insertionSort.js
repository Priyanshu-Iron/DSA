let arr = [4,7,9,1,3]

function insertionSort(arr,n){
    for(let i=0;i<=n-1;i++){
        let j=i;
        while(j>0&&arr[j-1]>arr[j]){
            let temp = arr[j-1];
            arr[j-1] = arr[j];
            arr[j] = temp;
            
            j--;
        }
    }
}

insertionSort(arr,arr.length)
console.log(arr);