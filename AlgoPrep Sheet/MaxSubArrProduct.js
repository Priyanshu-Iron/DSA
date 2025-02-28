const nums = [2, 3, -2, 4];

function maxProduct1(array) {
  let result = array[0];

  for (let i = 0; i < array.length; i++) {
    let currentProduct = 1;

    for (let j = i; j < array.length; j++) {
        currentProduct = currentProduct*array[j]

        if (currentProduct<0) {
            currentProduct = 1
        }

        result = Math.max(result,currentProduct)
    }
  }

  return result
}

function maxProduct2(array) {
    let result = array[0];
    let maxProduct = array[0];
    let minProduct = array[0];
  
    for (let i = 1; i < array.length; i++) {
      if (array[i] < 0) {
        [maxProduct, minProduct] = [minProduct, maxProduct];
      }
  
      maxProduct = Math.max(array[i], maxProduct * array[i]);
      minProduct = Math.min(array[i], minProduct * array[i]);
  
      result = Math.max(result, maxProduct);
    }
  
    return result;
}

console.log(maxProduct1(nums))

function maxSubArrayProduct(array) {}