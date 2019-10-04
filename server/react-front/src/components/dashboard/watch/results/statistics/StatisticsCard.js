import React from "react";
import * as PropTypes from "prop-types";
import Typography from "@material-ui/core/Typography";
import RoundCornerAvatar from "../../../../common/RoundCornerAvatar";
import Grid from "@material-ui/core/Grid";
import withStyles from "@material-ui/core/styles/withStyles";
import Card from "@material-ui/core/Card";
import CardHeader from "@material-ui/core/CardHeader";
import CardContent from "@material-ui/core//CardContent";

const styles = {
    cardRoot: {},
    grid: {
        width: 96,
    },
    roundCornerAvatar: {
        width: 80,
        height: 80,
    },
    text: {
        textOverflow: "ellipsis",
    },
};

const StatisticsCard = ({classes, models, scenes, frame}) => (
    <React.Fragment>
        <ModelCard classes={classes} models={models}/>
        <SceneCard classes={classes} scenes={scenes} frame={frame}/>
    </React.Fragment>
);

const ModelCard = ({classes, models}) => (
    <Card className={classes.cardRoot} elevation={0}>
        <CardHeader title="Analytic Model"/>
        <CardContent>
            <Grid container spacing={8}>
                {models.map((model, index) => {
                        const hashCode = model.hashCode().toString();
                        return <Grid key={index} className={classes.grid} item>
                            <RoundCornerAvatar classes={{root: classes.roundCornerAvatar}}
                                               src={`https://picsum.photos/144?image=${hashCode.substring(hashCode.length - 2)}`}/>
                            <Typography className={classes.text} align="center">
                                {model}
                            </Typography>
                        </Grid>;
                    }
                )}
            </Grid>
        </CardContent>
    </Card>
);

const SceneCard = ({classes, scenes, frame}) => {
    const sceneNum = scenes.findIndex(scene => scene.start <= frame && scene.end >= frame);
    if (sceneNum !== -1)
        return (
            <Card className={classes.cardRoot} elevation={0}>
                <CardHeader title="Result statistics" subheader={`Scene ${sceneNum + 1}`}/>
                <CardContent>
                    <Grid container spacing={8}>
                        {scenes[sceneNum].results.map(scene => {
                                const hashCode = scene.name.hashCode().toString();
                                return <Grid key={scene.name} item>
                                    <RoundCornerAvatar classes={{root: classes.roundCornerAvatar}}
                                                       src={scene.src || `https://picsum.photos/144?image=${hashCode.substring(hashCode.length - 3)}`}/>
                                    <Typography align="center" noWrap>{scene.name}</Typography>
                                    <Typography align="center">{scene.appearance}</Typography>
                                </Grid>;
                            }
                        )}
                    </Grid>
                </CardContent>
            </Card>
        );
    return null;
};

StatisticsCard.propTypes = {
    classes: PropTypes.object.isRequired,
    models: PropTypes.array.isRequired,
    scenes: PropTypes.array.isRequired,
};

export default withStyles(styles)(StatisticsCard);
