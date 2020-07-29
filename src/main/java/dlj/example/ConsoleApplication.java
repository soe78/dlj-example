package dlj.example;
/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import javax.annotation.Resource;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.Supplier;

@SpringBootApplication
public class ConsoleApplication implements CommandLineRunner {

	private static Logger LOG = LoggerFactory.getLogger(ConsoleApplication.class);

	/**
	 * Note: @{@link Autowired} will fail on matching the generic type here. To wire
	 * with Autowired and generic types consider the following: <code>
	 *     &#64;Autowired
	 *     Supplier<Predictor <?, ?>> autowiredProvider;
	 * </code>
	 *
	 * Then casting to the right type.
	 */
	@Resource
	private Supplier<Predictor<BufferedImage, DetectedObjects>> predictorProvider;

	public static void main(String[] args) {
		SpringApplication.run(ConsoleApplication.class, args);
	}

	@Override
	public void run(String... args) throws Exception {
		try (Predictor<BufferedImage, DetectedObjects> predictor = predictorProvider.get()) {
			DetectedObjects results = predictor.predict(loadImage("/puppy-in-white-and-red-polka.jpg"));
			results.items().stream().filter(r -> r.getProbability() > 0.8d).forEach(System.out::println);

			results = predictor.predict(loadImage("/cat.jpg"));
			results.items().stream().filter(r -> r.getProbability() > 0.8d).forEach(System.out::println);
		}
	}

	private BufferedImage loadImage(String name) throws IOException {
		return ImageIO.read(this.getClass().getResourceAsStream(name));
	}
}
