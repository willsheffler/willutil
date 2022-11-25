import willutil as wu

def main():
   test_rbsample_oct()

def test_rbsample_oct():
   samp = wu.search.RBSampler(cartsd=1, level=1, scale=10)

if __name__ == '__main__':
   main()