import React from "react";
import * as PropTypes from "prop-types";
import Grid from "@material-ui/core/Grid";
import withStyles from "@material-ui/core/styles/withStyles";
import DetectionCard from "./results/visual/DetectionCard";
import Switch from "@material-ui/core/es/Switch/Switch";
import FormControl from "@material-ui/core/es/FormControl/FormControl";
import FormControlLabel from "@material-ui/core/es/FormControlLabel/FormControlLabel";
import Player from "../common/player/Player";
import {injectParentState} from "../../../utils/utils";
import AudioCard from "./AudioCard";
import AudioDetectionCard from "./results/audio/DetectionCard";
import StatisticsCard from "./results/statistics/StatisticsCard";
import classNames from "classnames";

const styles = theme => ({
    root: {
        margin: "auto",
    },
    secondary: {
        position: "relative",
        overflow: "hidden",
    },
    detectionCardContainer: {
        position: "relative",
        overflow: "hidden",
    },
    cardRoot: {
        marginBottom: 2 * theme.spacing.unit,
    },
    detectionContainer: {
        "&[aria-disabled=true]": {
            display: "none",
        }
    }
});

class WatchApp extends React.PureComponent {
    state = {
        box: false,
        audio: false,
        frame: 0,
        time: 0,
        height: 0,
        events: {
            play: false,
            progress: 0,
        },
    };

    handleChange = name => event => {
        this.setState({[name]: event.target.checked});
    };

    handleResultToBoxes = () => {
        const {result = {}} = this.props.video;
        const {visual = {}} = result;

        return Object.keys(visual).filter(key => key !== "scene")
            .map(key => (visual[key][this.state.frame] || []))
            .reduce((x, y) => x.concat(y), [])
            .map(o => ({
                ...o,
                left: `${o.x1 * 100}%`,
                right: `${100 - o.x2 * 100}%`,
                top: `${o.y1 * 100}%`,
                bottom: `${100 - o.y2 * 100}%`,
            }));
    };

    handleProductToAds = () => {
        const {time} = this.state;
        const {products = []} = this.props.video;

        const ads = products.filter(product => {
                return product.start <= time && time <= product.end;
            }
        );

        return ads.length ? ads[Math.floor(Math.random() * ads.length)] : {};
    };

    componentWillMount() {
        const {location, fetchVideoInfo} = this.props;
        const params = new URLSearchParams(location.search);

        fetchVideoInfo(params.get("v"));
    }

    componentWillUnmount() {
        this.props.clearVideoInfo();
    }

    handleAudioWaveformLoaded = () => {
        this.setState({audio: true});
    };

    render() {
        const {classes, video} = this.props;
        const {box, audio, frame, height, events} = this.state;
        const {result = {}, audio: audioUrl, models = [], processed} = video;
        const {visual = {}, audio: audioResult = [], statistics = []} = result;

        return (
            <Grid className={classes.root} direction={'row'} container spacing={24}>
                <Grid item xs={12} md={8} lg={9}>
                    <Player boxOn={box}
                            video={video}
                            boxes={this.handleResultToBoxes()}
                            ads={this.handleProductToAds()}
                            handleUpdateParentState={injectParentState(this, this.setState,
                                ["time", "frame", "height", "events"])}
                    />

                    <FormControl>
                        <FormControlLabel
                            control={<Switch value="box" onChange={this.handleChange("box")} checked={box}/>}
                            label={"Detection Box"}
                            disabled={!processed}
                        />
                    </FormControl>
                    <FormControl>
                        <FormControlLabel
                            control={<Switch value="audio" onChange={this.handleChange("audio")} checked={audio}/>}
                            label={"Audio Waveform"}
                            disabled={!audioUrl}
                        />
                    </FormControl>

                    <AudioCard on={audio} audio={audioUrl} events={events} onReady={this.handleAudioWaveformLoaded}/>
                    <div
                        className={classNames(classes.detectionCardContainer, classes.cardRoot, classes.detectionContainer)}
                        aria-disabled={!processed}>
                        {/* TODO: solve the mapper */}
                        <AudioDetectionCard result={audioResult}
                                            frame={Math.round(frame * audioResult.length / video.totalFrame)}/>
                    </div>

                </Grid>
                <Grid className={classNames(classes.secondary, classes.detectionContainer)} item xs={12} md={4} lg={3}
                      aria-disabled={!processed}
                >
                    <div className={classNames(classes.detectionCardContainer, classes.cardRoot)}
                         style={{height: height}}>
                        <DetectionCard result={visual} frame={frame}/>
                    </div>
                    <StatisticsCard classes={{cardRoot: classes.cardRoot}} scenes={statistics} models={models}
                                    frame={frame}/>
                </Grid>
            </Grid>
        );
    }
}

WatchApp.propTypes = {
    classes: PropTypes.object.isRequired,
    video: PropTypes.shape({
        quality: PropTypes.array,
        defaultQuality: PropTypes.number,
        url: PropTypes.string,
        pic: PropTypes.string,
        result: PropTypes.object,
        ad: PropTypes.array,
    }).isRequired,
    previewedAd: PropTypes.object,
    fetchVideoInfo: PropTypes.func.isRequired,
    clearVideoInfo: PropTypes.func.isRequired,
};

export default withStyles(styles)(WatchApp);
