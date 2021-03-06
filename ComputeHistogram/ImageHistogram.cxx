#include "itkImageToListSampleAdaptor.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkHistogram.h"
#include "itkSampleToHistogramFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"

typedef float               PixelType;
const unsigned int          Dimension = 3;

typedef itk::Image<PixelType, Dimension > ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;

int main( int argc, char * argv [] )
{

  if( argc < 2 )
  {
    std::cerr << "Missing command line arguments" << std::endl;
    std::cerr << "Usage : " << argv[0] << " inputImageFileName [Partitions-per-dimension]" << std::endl;
    return -1;
  }

  std::vector<unsigned int> NPartitions;
  if( argc > 2 )
  {
    for(unsigned int c=2; c<argc; c++)
    {
      NPartitions.push_back(atoi(argv[c]));
    }
    if( NPartitions.size() == 1)
      NPartitions.assign( Dimension, NPartitions[0]);
    else if( NPartitions.size() != Dimension )
    {
      std::cerr << "Number of partitions per dimension and image dimension mismatch." << std::endl;
      return -1;
    }
  }

  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName( argv[1] );
  typedef itk::Statistics::ImageToListSampleAdaptor< ImageType >   AdaptorType;

  try
    {
    reader->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Problem reading image file : " << argv[1] << std::endl;
    std::cerr << excp << std::endl;
    return -1;
    }

  unsigned int NBlocks = 1;
  std::vector<unsigned int> blkSize(Dimension), cumSize(Dimension);
  cumSize[0] = 1;
  ImageType::SizeType imgSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  for (unsigned int d = 0; d<Dimension; d++ )
  {
    blkSize[d] = imgSize[d]/NPartitions[d];
    NBlocks = NBlocks*NPartitions[d];
    if(d>0)
      cumSize[d] = cumSize[d-1]*NPartitions[d-1];
  }

  typedef itk::MinimumMaximumImageCalculator< ImageType > MinMaxCalculatorType;
  MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
  minMaxCalculator->SetImage(reader->GetOutput());
  minMaxCalculator->Compute();
  ImageType::PixelType imgMin = minMaxCalculator->GetMinimum();
  ImageType::PixelType imgMax = minMaxCalculator->GetMaximum();
  float range = static_cast< float >( imgMax-imgMin );

  //Build the block partition image:
  ImageType::Pointer partitions = ImageType::New();
  partitions->SetRegions( reader->GetOutput()->GetLargestPossibleRegion() );
  partitions->SetOrigin( reader->GetOutput()->GetOrigin() );
  partitions->SetSpacing( reader->GetOutput()->GetSpacing() );
  partitions->SetDirection( reader->GetOutput()->GetDirection() );
  partitions->Allocate();
  itk::ImageRegionIteratorWithIndex< ImageType > it(partitions,partitions->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator< ImageType > imgIt(reader->GetOutput(),reader->GetOutput()->GetLargestPossibleRegion());
  for( imgIt.GoToBegin(), it.GoToBegin(); !it.IsAtEnd(); ++it, ++imgIt)
  {
    ImageType::IndexType idx = it.GetIndex();
    unsigned int pos = 0;
    for(unsigned int d=0; d<Dimension; d++)
    {
      idx[d] = idx[d]/blkSize[d];
      idx[d] = (idx[d]>NPartitions[d]-1)?(NPartitions[d]-1):idx[d];
      pos += idx[d]*cumSize[d];
    }
    it.Set( float(pos)*range + imgIt.Get() );
  }
  /*
  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
  writer->SetFileName("img.nii");
  writer->SetInput( partitions );
  writer->Update();
  */
  //  std::cout << imgMin << "\t" << imgMax << "\t" << range << std::endl;

  AdaptorType::Pointer adaptor = AdaptorType::New();
  adaptor->SetImage( partitions );
  typedef PixelType HistogramMeasurementType;
  typedef itk::Statistics::Histogram< HistogramMeasurementType >
    HistogramType;
  const unsigned int numberOfComponents = 1;
  HistogramType::SizeType size( numberOfComponents );
  size.Fill( 255*NBlocks );
  typedef itk::Statistics::SampleToHistogramFilter<
                                                AdaptorType,
                                                HistogramType>
                                                FilterType;
  FilterType::Pointer filter = FilterType::New();

  filter->SetInput( adaptor );
  filter->SetHistogramSize( size );
  filter->SetMarginalScale( 10 );

  HistogramType::MeasurementVectorType min( numberOfComponents );
  HistogramType::MeasurementVectorType max( numberOfComponents );
  min.Fill( imgMin );
  max.Fill( imgMax + (NBlocks-1)*range );

  filter->SetHistogramBinMinimum( min );
  filter->SetHistogramBinMaximum( max );
  filter->Update();

  HistogramType::ConstPointer histogram = filter->GetOutput();
  const unsigned int histogramSize = histogram->Size();

  for( unsigned int bin=0; bin < histogramSize; bin++ )
  {
    std::cout << bin << "," << histogram->GetFrequency( bin, 0 ) << std::endl;
  }

  return 0;
}
