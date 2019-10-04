import { connect } from 'react-redux';
import VideoManagement from './VideoManagement';
import {
  VIDEO_delete,
  VIDEOgetList,
  VIDEO_update,
} from '../../../redux/actions/video';

const mapStateToProps = state => ({
  videos: state.video.videoList.videos,
});

const mapDispatchToProps = dispatch => ({
  handleFetchVideos: () => dispatch(VIDEOgetList()),
  handleProcessVideo: id => dispatch(VIDEO_update(id)),
  handleDeleteVideo: (id, successCallback) =>
    dispatch(VIDEO_delete(id, successCallback)),
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(VideoManagement);
