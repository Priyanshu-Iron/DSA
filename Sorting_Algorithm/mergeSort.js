// Merge sort Algorithms:

function Mergesort(arr,low,high){

    if(low >=high){
        return;
    }
    
    let mid = Math.floor((low + high) / 2);
    Mergesort(arr,low,mid);
    Mergesort(arr,mid+1,high);
    Mergesort(arr,low,mid,high);
     Merge(arr, low, mid, high);
    
    }
    
    
    function Merge(arr,low,mid,high){
     let temp = [];
     let left = low;
     let right = mid+1;
     while(left<=mid && right<=high){ 
     
         if(arr[left]<=arr[right]){ 
            temp.push(arr[left]);
            left++
    
    }
    else{
        temp.push(arr[right]);
        right++;
    }
    
     }
    
    
    while(left<=mid){
        temp.push(arr[left]);
        left++;
    }
    while(right<=high){
        temp.push(arr[right]);
        right++;
    }
    for(let i = low; i<=high;i++){
        arr[i] = temp[i-low];
    }
    
     }
    
    let arr = [3,1,2,3,1,5,6,2,4];
    let n = arr.length;
    Mergesort(arr,0,n-1);
    console.log(arr);
    
    
    // Tc of this code will be -> O(nlogn)
    // sc will be O(N)