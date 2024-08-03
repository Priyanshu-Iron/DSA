const arr = [4,25,66,731,1,3]

function bubbleSort(arr,n){
    for(let i=n-1;i>=0;i--){
        let didSwap = 0
        for(let j=0;j<=i-1;j++){
            if(arr[j]>arr[j+1]){
            let temp = arr[j+1]
            arr[j+1]=arr[j];
            arr[j]=temp;
            didSwap = 1
            }
        }
        if (didSwap == 0) {
            break;
        }
        // console.log("runs");
        
    }
}

bubbleSort(arr,arr.length)
console.log(arr);