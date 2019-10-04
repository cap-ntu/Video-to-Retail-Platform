import React from 'react';
import * as PropTypes from 'prop-types';
import Grid from "@material-ui/core/Grid";
import VideoItemCard from "./VideoItemCard";
import withStyles from "@material-ui/core/styles/withStyles";
import Typography from "@material-ui/core/Typography";
import Fade from "@material-ui/core/Fade";

const styles = {
    root: {
        width: "85%",
        margin: "auto",
    }
};

/**
 * videos video format: [{name: string, cover: string, id: string}]
 */
class VideoManagement extends React.PureComponent {

    componentDidMount() {
        this.props.handleFetchVideos();
    }

    handleDeleteVideo = id => () => {
        const {handleFetchVideos, handleDeleteVideo} = this.props;
        handleDeleteVideo(id, handleFetchVideos);
    };

    handleProcessVideo = id => () => {
        const {handleFetchVideos, handleProcessVideo} = this.props;
        handleProcessVideo(id, handleFetchVideos);
        handleFetchVideos();
    };

    render() {
        const {classes, videos} = this.props;

        return (
            <Fade in={true}>
                <div>
                    {
                        videos.length ?
                            <Grid className={classes.root} container spacing={16}> {
                                videos.map(video =>
                                    <Grid key={video.id} item>
                                        <VideoItemCard id={video.id.toString()}
                                                       name={video.name}
                                                       cover={video.cover}
                                                       processed={video.processed}
                                                       handleProcess={this.handleProcessVideo(video.id)}
                                                       handleDelete={this.handleDeleteVideo(video.id)}
                                                       disable={video.isProcessing}
                                        />
                                    </Grid>)}
                            </Grid> :
                            <Typography align={"center"}>
                                There is no uploaded video viewed by you currently.
                            </Typography>
                    }
                </div>
            </Fade>
        );
    }
}

VideoManagement.defaultProps = {
    videos: [],
};

VideoManagement.propTypes = {
    classes: PropTypes.object.isRequired,
    videos: PropTypes.arrayOf(
        PropTypes.shape({
            id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
            name: PropTypes.string.isRequired,
            cover: PropTypes.string,
        }).isRequired,
    ),
    handleFetchVideos: PropTypes.func.isRequired,
    handleProcessVideo: PropTypes.func.isRequired,
    handleDeleteVideo: PropTypes.func.isRequired,
};

export default withStyles(styles)(VideoManagement);
