import h5py

x = 0.0
delta = 0.1

# Export data   
for d in [1.0*0.2*i for i in range(11)]:
    for p in range(128):

        with h5py.File('dataN36/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(d,x,delta,p),'r') as hf:
            H2_initial = hf.get('H2_initial')
            H2_diag = hf.get('H2_diag')
            Hint = hf.get('Hint')
            flow2 = hf.get('flow2')
            flow4 = hf.get('flow4')

        with h5py.File('dataN36/tflow-d%.2f-x%.2f-Jz%.2f-p%s.h5' %(d,x,delta,p),'w') as hf:
            hf.create_dataset('H2_initial',data=H2_initial)
            hf.create_dataset('H2_diag',data=H2_diag)
            hf.create_dataset('Hint',data=Hint)
            hf.create_dataset('flow2',data=flow2)
            hf.create_dataset('flow4',data=flow4)

